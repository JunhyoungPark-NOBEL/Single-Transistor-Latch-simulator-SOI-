<#
STL Simulator: local installer and control script for Windows (Windows PowerShell 5.1 or later).
Saved as UTF-8 with BOM so that Windows PowerShell 5.1 reads the Korean text correctly.

  From the installer kit (install-windows.bat):  install-windows.ps1 install [options]
  Installed copy (bin\*.bat):  stl-sim.ps1 start|stop|restart|open|status|logs|rollback|update|uninstall [options]
  Options: -Yes  -Port N  -InstallDir DIR  -PasswordStdin  -NoBrowser  -NoShortcut  -RemoveCache  -KeepCache
           -RemoveBuildCache  -KeepBuildCache  -Rollback  -Kit DIR  -Pause  -PauseOnError
           (the bash spellings --yes, --port N, ... work too)
  Environment: STL_INSTALL_PASSWORD  password for non-interactive installs (removed from the environment at once)
               STL_BUILD_EXTRA_ARGS  extra "docker build" arguments (e.g. --build-context for a proxy CA image)

Binary data (the encrypted payload and the decrypted tar stream) never passes through a PowerShell pipeline:
Windows PowerShell 5.1 re-encodes native-command pipes as text and would corrupt it. stl-pipe.cmd (cmd.exe:
byte-exact pipes and "<" redirection) runs  docker run -i ... < payload | docker build -  so the plaintext source
goes from the decrypting container's stdout straight into docker build and is never written to disk; afterwards the
installer deletes BuildKit's cached copy of that build context. The password is handed over in the environment
variable STL_PW (docker run -e STL_PW, by name only: never on a command line) and removed right after. (cmd.exe cannot
put a Unicode password on a pipe unchanged, so unlike macOS/Linux it is not sent on stdin; while the 1-2 s helper
container runs, users who may use Docker can see it with docker inspect.)
Exit codes: 0 ok, 20 an error this script explained (install-windows.bat pauses only for other codes).
#>

$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$Container = 'stl-simulator'
$Repo = 'stl-simulator'
$Volume = 'stl-simulator-cache'
$HelperImage = 'python:3.11-slim'
$Label = 'org.stl-simulator'
$PortMin = 8000
$PortMax = 8010
$EngineTimeout = 180
$HealthTimeout = 240
$DockerDownloadUrl = 'https://www.docker.com/products/docker-desktop/'
$StateKeys = @('PORT', 'CURRENT_VERSION', 'CURRENT_IMAGE', 'CURRENT_BUILD', 'PREVIOUS_VERSION', 'PREVIOUS_IMAGE',
               'WORKERS', 'INSTALLED')
$ExitExplained = 20
$Commands = @('install', 'start', 'stop', 'restart', 'open', 'status', 'logs', 'rollback', 'update', 'uninstall', 'help')

$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$KitMode = Test-Path -LiteralPath (Join-Path $ScriptDir 'kit-info.txt')
$ScriptArgs = @($args)

$script:Opt = @{ Yes = $false; Port = ''; InstallDir = ''; PasswordStdin = $false; NoBrowser = $false
                 NoShortcut = $false; Cache = ''; BuildCache = ''; Rollback = $false; Kit = ''; Pause = $false
                 PauseOnError = $false }
$script:Command = 'help'
if ($KitMode) { $script:Command = 'install' }
$script:LogFile = $null
$script:TmpLog = $null
$script:Pw = $null
$script:InstallDir = $null
$script:S = @{}
$script:Kit = @{}
$script:DkCode = 0
$script:Workers = 1
$script:PipeDir = $ScriptDir
$script:TmpKit = $null

# ----------------------------------------------------------------------------------------------- output
function Write-Log([string]$Text) {
    if (-not $script:LogFile) { return }
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $Text + "`r`n"
    try { [System.IO.File]::AppendAllText($script:LogFile, $line, (New-Object System.Text.UTF8Encoding($false))) } catch { }
}

function Say([string]$Ko, [string]$En = '') {
    Write-Host $Ko
    if ($En) { Write-Host ('  ' + $En) -ForegroundColor DarkGray }
    if ($En) { Write-Log ($Ko + ' | ' + $En) } else { Write-Log $Ko }
}

function Step([string]$Ko, [string]$En) {
    Write-Host ''
    Write-Host ('==> ' + $Ko) -ForegroundColor Cyan
    Write-Host ('    ' + $En) -ForegroundColor DarkGray
    Write-Log ('== ' + $Ko + ' | ' + $En)
}

function Warn([string]$Ko, [string]$En = '') {
    Write-Host ('[주의] ' + $Ko) -ForegroundColor Yellow
    if ($En) { Write-Host ('       ' + $En) -ForegroundColor Yellow }
    Write-Log ('WARN ' + $Ko + ' | ' + $En)
}

function Test-Interactive {
    try { return ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) } catch { return $false }
}

function Pause-End([switch]$Force) {
    if (($script:Opt.Pause -or $Force) -and (Test-Interactive)) {
        Write-Host ''
        [void](Read-Host 'Enter를 누르면 창을 닫습니다 (press Enter to close)')
    }
}

function Clear-Secret {
    $script:Pw = $null
    Remove-Item Env:\STL_PW -ErrorAction SilentlyContinue
}

function Remove-TmpLog {
    if ($script:TmpLog -and (Test-Path -LiteralPath $script:TmpLog)) {
        Remove-Item -LiteralPath $script:TmpLog -Force -ErrorAction SilentlyContinue
    }
}

function Remove-TmpKit {
    if ($script:TmpKit -and (Test-Path -LiteralPath $script:TmpKit)) {
        Remove-Item -LiteralPath $script:TmpKit -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Die([string]$Ko, [string]$En = '') {
    Write-Host ''
    Write-Host ('[오류] ' + $Ko) -ForegroundColor Red
    if ($En) { Write-Host ('       ' + $En) -ForegroundColor Red }
    Write-Log ('ERROR ' + $Ko + ' | ' + $En)
    if ($script:LogFile -and ($script:LogFile -ne $script:TmpLog)) { Write-Host ('       로그 / log: ' + $script:LogFile) }
    Clear-Secret
    Remove-TmpLog
    Remove-TmpKit
    if ($script:Opt.Pause -or $script:Opt.PauseOnError) { Pause-End -Force }
    exit $ExitExplained
}

# -Strict (installing system software): only an answer typed by a person counts as "yes" (not -Yes, not a missing console).
function Ask([string]$Ko, [string]$En, [bool]$Default, [switch]$Strict) {
    if ($Strict) { if (-not (Test-Interactive)) { return $false } }
    elseif ($script:Opt.Yes) { return $true }
    if (-not (Test-Interactive)) { return $Default }
    $hint = '[y/N]'
    if ($Default) { $hint = '[Y/n]' }
    Write-Host $Ko
    $ans = Read-Host ('  ' + $En + ' ' + $hint)
    if ($ans -match '^\s*(y|yes|네|예|응)') { return $true }
    if ($ans -match '^\s*(n|no|아니)') { return $false }
    return $Default
}

# ----------------------------------------------------------------------------------------------- helpers
function Get-KeyValueFile([string]$Path) {
    $h = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $h }
    foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
        $i = $line.IndexOf('=')
        if ($i -gt 0) { $h[$line.Substring(0, $i).Trim()] = $line.Substring($i + 1).Trim() }
    }
    return $h
}

function Write-TextFile([string]$Path, [string]$Text, [string]$Kind = 'bom') {
    $t = ($Text -replace "`r`n", "`n") -replace "`n", "`r`n"
    if ($Kind -eq 'ascii') { [System.IO.File]::WriteAllText($Path, $t, [System.Text.Encoding]::ASCII) }
    else { [System.IO.File]::WriteAllText($Path, $t, (New-Object System.Text.UTF8Encoding($true))) }
}

function Get-Sha256([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $fs = [System.IO.File]::OpenRead($Path)
    try { $hash = $sha.ComputeHash($fs) } finally { $fs.Close(); $sha.Dispose() }
    return (($hash | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Get-Local([string]$Url) {
    try {
        $req = [System.Net.WebRequest]::Create($Url)
        $req.Proxy = $null
        $req.Timeout = 5000
        $resp = $req.GetResponse()
        try {
            $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
            return $sr.ReadToEnd()
        } finally { $resp.Close() }
    } catch { return '' }
}

function Test-PortInUse([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(400) -and $c.Connected) { return $true }
        return $false
    } catch { return $false } finally { $c.Close() }
}

function Open-Url([string]$Url) {
    try { Start-Process $Url; return $true } catch { return $false }
}

function Get-Stamp { return (Get-Date -Format 'yyyyMMdd-HHmmss') }

function Compute-Workers([int]$Ncpu, [long]$MemBytes) {
    $w = $Ncpu - 1
    if ($w -gt 8) { $w = 8 }
    $memMb = [long]($MemBytes / 1MB)
    if ($memMb -gt 0) {
        $byMem = [int][Math]::Floor(($memMb - 1024) / 300)
        if ($w -gt $byMem) { $w = $byMem }
    }
    if ($w -lt 1) { $w = 1 }
    return $w
}

# ----------------------------------------------------------------------------------------------- docker
# Dk <docker arguments...>: runs docker, returns its output (stdout+stderr) as one string, exit code in $script:DkCode
function Dk {
    $o = & docker @args 2>&1 | ForEach-Object { "$_" }
    $script:DkCode = $LASTEXITCODE
    return ($o -join "`n")
}

function Find-Docker {
    if (Get-Command docker -ErrorAction SilentlyContinue) { return $true }
    foreach ($d in @((Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin'),
                     (Join-Path $env:LOCALAPPDATA 'Programs\Docker\Docker\resources\bin'))) {
        if (Test-Path -LiteralPath (Join-Path $d 'docker.exe')) { $env:Path = $d + ';' + $env:Path; return $true }
    }
    return $false
}

function Get-DockerDesktopExe {
    foreach ($p in @((Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
                     (Join-Path $env:LOCALAPPDATA 'Programs\Docker\Docker\Docker Desktop.exe'))) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Offer-DockerInstall {
    Say 'Docker Desktop이 설치되어 있지 않습니다. STL Simulator는 Docker 안에서 실행됩니다.' `
        'Docker Desktop is not installed. The simulator runs inside Docker.'
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        if (Ask 'winget으로 Docker Desktop을 설치할까요? (관리자 승인 창이 뜰 수 있습니다)' 'Install Docker Desktop with winget?' $true -Strict) {
            & winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                Say 'winget 설치가 끝나지 않았습니다. 내려받기 페이지를 엽니다.' 'winget did not finish; opening the download page.'
                [void](Open-Url $DockerDownloadUrl)
            }
            Say '설치가 끝나면 (필요하면 재부팅 후) Docker Desktop을 한 번 실행해 약관에 동의하고, 이 설치 파일을 다시 실행하세요.' `
                'When it is installed (reboot if asked), start Docker Desktop once, accept its terms, then run this installer again.'
            Pause-End -Force
            exit $ExitExplained
        }
    }
    Say 'Docker Desktop 내려받기 페이지를 엽니다. 설치 후 한 번 실행한 다음 이 설치 파일을 다시 실행하세요.' `
        'Opening the Docker Desktop download page; install it, start it once, then run this installer again.'
    if (-not (Open-Url $DockerDownloadUrl)) { Say ('  ' + $DockerDownloadUrl) }
    Pause-End -Force
    exit $ExitExplained
}

function Test-Engine {
    [void](Dk info --format '{{.ServerVersion}}')
    return ($script:DkCode -eq 0)
}

function Explain-Wsl([string]$EngineError) {
    $hv = $null; $vt = $null
    try { $hv = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).HypervisorPresent } catch { }
    try { $vt = (Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1).VirtualizationFirmwareEnabled } catch { }
    $wslOk = $false
    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
        & wsl.exe --status *> $null
        $wslOk = ($LASTEXITCODE -eq 0)
    }
    Write-Log ("engine error: $EngineError / hypervisor=$hv vt=$vt wsl=$wslOk")
    if (($hv -eq $false) -and ($vt -eq $false)) {
        Say '- CPU 가상화가 꺼져 있습니다. BIOS/UEFI 설정에서 Intel VT-x 또는 AMD SVM(가상화)을 켜야 합니다.' `
            '- CPU virtualization is off: enable Intel VT-x / AMD SVM in the BIOS/UEFI settings.'
    }
    if ((-not $wslOk) -or ($EngineError -match 'WSL|wsl')) {
        Say '- WSL2가 설치되지 않았거나 오래되었습니다. 시작 메뉴에서 PowerShell을 "관리자 권한으로 실행"한 뒤' `
            '- WSL2 is missing or outdated. In an administrator PowerShell run:'
        Say '    wsl --install        (이미 있으면: wsl --update)   → 컴퓨터를 다시 시작하세요.' '    wsl --install   (or: wsl --update), then restart the computer.'
    }
    if ($EngineError -match 'Access is denied|access denied|permission') {
        Say '- 이 사용자가 docker-users 그룹에 없습니다. 관리자 PowerShell에서: net localgroup docker-users "사용자이름" /add → 로그아웃 후 다시 로그인' `
            '- Add your account to the docker-users group (administrator), then sign out and in again.'
    }
    Say '- Docker Desktop 창에 약관 동의, 업데이트, 오류 안내가 떠 있는지 확인하세요.' `
        '- Check the Docker Desktop window for a license, update or error message.'
}

function Ensure-Engine {
    if (Test-Engine) { return }
    $err = Dk info --format '{{.ServerVersion}}'
    if ($err -match 'Access is denied|access denied|permission denied') {
        # not in the docker-users group: waiting cannot fix this
        Explain-Wsl $err
        Die 'Docker에 접근할 권한이 없습니다 (docker-users 그룹).' "Access to Docker is denied (docker-users group): $err"
    }
    Say 'Docker 엔진이 꺼져 있어 Docker Desktop을 시작합니다...' 'The Docker engine is not running; starting Docker Desktop...'
    $exe = Get-DockerDesktopExe
    if ($exe) { try { Start-Process -FilePath $exe } catch { } }
    else { Warn 'Docker Desktop.exe를 찾지 못했습니다. 직접 실행해 주세요.' 'Docker Desktop.exe not found; please start it yourself.' }
    Say 'Docker Desktop 창이 뜨면 약관에 동의(Accept)하세요. 로그인(Sign in)은 건너뛰어도 됩니다. 처음에는 1–3분 걸립니다.' `
        'If the Docker Desktop window opens, accept its terms (signing in can be skipped); the first start takes 1-3 min.'
    $waited = 0
    $total = 0
    while (-not (Test-Engine)) {
        if ($waited -ge $EngineTimeout) {
            if ((-not $script:Opt.Yes) -and (Test-Interactive) -and
                (Ask 'Docker가 아직 준비되지 않았습니다. 더 기다릴까요? (WSL2 첫 시작은 오래 걸릴 수 있습니다)' 'Docker is not ready yet. Keep waiting?' $true)) {
                $waited = 0
                continue
            }
            $err = Dk info --format '{{.ServerVersion}}'
            Explain-Wsl $err
            Die ("Docker 엔진이 $total 초 안에 시작되지 않았습니다.") ("The Docker engine did not start within $total s: $err")
        }
        if (($total % 15) -eq 0) { Say ("  Docker 시작을 기다리는 중... ($total s)") '  waiting for Docker...' }
        Start-Sleep -Seconds 3
        $waited += 3
        $total += 3
    }
    Say 'Docker 엔진이 준비되었습니다.' 'Docker engine is ready.'
}

function Check-Platform {
    $ostype = (Dk info --format '{{.OSType}}').Trim()
    if ($ostype -eq 'windows') {
        Say 'Docker Desktop이 Windows 컨테이너 모드입니다. Linux 컨테이너 모드로 바꿔야 합니다.' 'Docker Desktop is in Windows-containers mode; Linux containers are needed.'
        $cli = Join-Path $env:ProgramFiles 'Docker\Docker\DockerCli.exe'
        if ((Test-Path -LiteralPath $cli) -and (Ask '지금 Linux 컨테이너 모드로 바꿀까요?' 'Switch to Linux containers now?' $true)) {
            & $cli -SwitchLinuxEngine
            Start-Sleep -Seconds 10
            Ensure-Engine
            $ostype = (Dk info --format '{{.OSType}}').Trim()
        }
    }
    if ($ostype -ne 'linux') {
        Die "Docker가 Linux 컨테이너 모드가 아닙니다 ($ostype). 작업 표시줄의 Docker 아이콘 → 'Switch to Linux containers'." 'Docker must run Linux containers.'
    }
    $arch = (Dk info --format '{{.Architecture}}').Trim()
    $ncpu = 2; $mem = [long]0
    [void][int]::TryParse((Dk info --format '{{.NCPU}}').Trim(), [ref]$ncpu)
    [void][long]::TryParse((Dk info --format '{{.MemTotal}}').Trim(), [ref]$mem)
    $script:Workers = Compute-Workers $ncpu $mem
    $memMb = [long]($mem / 1MB)
    Say ("Docker: $arch, CPU $ncpu, 메모리 $memMb MB → 계산 프로세스 $($script:Workers)개") `
        ("Docker: $arch, $ncpu CPUs, $memMb MB memory -> $($script:Workers) compute workers")
    if (($memMb -gt 0) -and ($memMb -lt 3500)) {
        Warn "Docker에 할당된 메모리가 적습니다 ($memMb MB). 4 GB 이상 권장 (WSL2는 %UserProfile%\.wslconfig 의 memory=)." `
             "Docker has little memory ($memMb MB); 4 GB or more is recommended."
    }
}

function Check-Disk([int]$NeedGb) {
    try {
        $qual = Split-Path -Qualifier $env:LOCALAPPDATA
        $drive = Get-PSDrive -Name ($qual.TrimEnd(':')) -ErrorAction Stop
        $freeGb = [Math]::Floor($drive.Free / 1GB)
    } catch { return }
    if ($freeGb -lt $NeedGb) {
        Warn "$qual 드라이브의 여유 공간이 $freeGb GB입니다. 설치에는 약 $NeedGb GB가 필요합니다." "Only $freeGb GB free on $qual; about $NeedGb GB are needed."
        if (-not (Ask '그래도 계속할까요?' 'Continue anyway?' $false)) { Die '디스크 공간을 확보한 뒤 다시 실행하세요.' 'Free some disk space and run again.' }
    }
}

function Test-ContainerExists { [void](Dk container inspect $Container); return ($script:DkCode -eq 0) }
function Test-ContainerRunning { return ((Dk container inspect -f '{{.State.Status}}' $Container).Trim() -eq 'running') }
function Test-Image([string]$Image) {
    if (-not $Image) { return $false }
    [void](Dk image inspect $Image)
    return ($script:DkCode -eq 0)
}

# ----------------------------------------------------------------------------------------------- state
function Load-State {
    $script:S = Get-KeyValueFile (Join-Path $script:InstallDir 'install-state.txt')
    foreach ($k in $StateKeys) { if (-not $script:S.ContainsKey($k)) { $script:S[$k] = '' } }
}

function Save-State {
    $lines = @()
    foreach ($k in $StateKeys) { $lines += ($k + '=' + [string]$script:S[$k]) }
    Write-TextFile (Join-Path $script:InstallDir 'install-state.txt') (($lines -join "`n") + "`n") 'bom'
}

$Marker = '.stl-simulator-folder'   # written into the install folder as soon as it is created
function Test-OurDir([string]$Dir) {
    if (-not $Dir) { return $false }
    return ((Test-Path -LiteralPath (Join-Path $Dir $Marker)) -or
            (Test-Path -LiteralPath (Join-Path $Dir 'install-state.txt')) -or
            (Test-Path -LiteralPath (Join-Path $Dir 'bin\stl-sim.ps1')))
}

# A typed or pasted folder: quotes removed, %VARIABLES% expanded, a leading ~ = the user profile.
function Normalize-Dir([string]$Dir) {
    $d = $Dir.Trim().Trim('"').Trim("'").Trim()
    $d = [Environment]::ExpandEnvironmentVariables($d)
    if ($d -eq '~') { $d = $env:USERPROFILE }
    elseif ($d -match '^~[\\/]') { $d = Join-Path $env:USERPROFILE $d.Substring(2) }
    return $d
}

function Resolve-InstallDir {
    $dir = $null
    if ($script:Opt.InstallDir) { $dir = Normalize-Dir $script:Opt.InstallDir }
    elseif ((-not $KitMode) -and (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $ScriptDir) 'install-state.txt'))) {
        $dir = Split-Path -Parent $ScriptDir
    } else {
        $labels = Dk container inspect -f '{{json .Config.Labels}}' $Container
        if ($script:DkCode -eq 0) {
            try {
                $obj = $labels | ConvertFrom-Json
                $prop = $obj.PSObject.Properties[$Label + '.dir']
                if ($prop -and (Test-OurDir $prop.Value)) { $dir = $prop.Value }
            } catch { }
        }
        if (-not $dir) { $dir = Join-Path $env:LOCALAPPDATA 'STL-Simulator' }
    }
    try { $dir = [System.IO.Path]::GetFullPath($dir).TrimEnd('\') }
    catch { Die "설치 폴더로 쓸 수 없는 경로입니다: $dir" 'Invalid install folder path.' }
    $bad = @($env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA, [System.IO.Path]::GetPathRoot($dir).TrimEnd('\'))
    foreach ($b in $bad) { if ($b -and ($dir -eq $b.TrimEnd('\'))) { Die "설치 폴더로 쓸 수 없는 경로입니다: $dir" 'Unsuitable install folder.' } }
    if (($dir -match '[\\/]~') -or ($dir -match '%')) {
        Die "경로에 '~' 또는 '%'로 된 폴더 이름이 있습니다: $dir (전체 경로로 입력하세요)" "The path contains a folder named with '~' or '%'; type the full path."
    }
    $script:InstallDir = $dir
}

# ----------------------------------------------------------------------------------------------- password + build
function Invoke-Pipe([string]$Mode) {
    # verify: returns the exit code.  build: streams the build output (filtered) and returns docker build's exit code.
    $env:STL_PW = $script:Pw
    $env:STL_HELPER_B64 = $script:HelperB64
    $env:STL_EXPECT_SHA256 = $script:Kit['ARCHIVE_SHA256']
    $env:STL_PAYLOAD = $script:Kit['PAYLOAD']
    $env:STL_HELPER_IMAGE = $HelperImage
    Push-Location -LiteralPath $script:PipeDir
    try {
        if ($Mode -eq 'verify') {
            $o = & cmd.exe /d /c .\stl-pipe.cmd verify 2>&1 | ForEach-Object { "$_" }
            $code = $LASTEXITCODE
            Write-Log ('verify: ' + ($o -join ' '))
            return $code
        }
        $sw = New-Object System.IO.StreamWriter($script:BuildLog, $false, (New-Object System.Text.UTF8Encoding($false)))
        $start = Get-Date
        $seen = @{}
        try {
            & cmd.exe /d /c .\stl-pipe.cmd build 2>&1 | ForEach-Object {
                $line = "$_"
                $sw.WriteLine($line)
                if ($line -match '\[warmup\]') {
                    Write-Host ('          ' + ($line -replace '^#\d+ [\d.]+ ', ''))
                } elseif ($line -match 'ERROR|error:') {
                    Write-Host ('  ' + $line) -ForegroundColor Yellow
                } elseif ($line -match '^(#\d+) \[[A-Za-z0-9_-]+ +\d+/\d+\] ') {
                    $id = $Matches[1]
                    if (-not $seen.ContainsKey($id)) {
                        $seen[$id] = $true
                        $el = (Get-Date) - $start
                        $txt = $line -replace '^#\d+ ', ''
                        if ($txt.Length -gt 110) { $txt = $txt.Substring(0, 110) }
                        Write-Host ('  [{0:00}:{1:00}] {2}' -f [int][Math]::Floor($el.TotalMinutes), $el.Seconds, $txt)
                    }
                }
            }
            return $LASTEXITCODE
        } finally { $sw.Close() }
    } finally {
        Pop-Location
        Remove-Item Env:\STL_PW, Env:\STL_HELPER_B64 -ErrorAction SilentlyContinue
    }
}

function Read-Password {
    $sec = Read-Host -AsSecureString '비밀번호를 입력하세요 (입력 내용은 보이지 않습니다) / Password'
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

function Get-Password {
    $max = 3
    $given = $null
    if ($env:STL_INSTALL_PASSWORD) { $given = $env:STL_INSTALL_PASSWORD; $max = 1 }
    elseif ($script:Opt.PasswordStdin) { $given = [Console]::In.ReadLine(); $max = 1 }
    Remove-Item Env:\STL_INSTALL_PASSWORD -ErrorAction SilentlyContinue
    for ($try = 1; $try -le $max; $try++) {
        if ($given) { $script:Pw = $given; $given = $null }
        else {
            if (-not (Test-Interactive)) { Die '비밀번호를 입력받을 수 없습니다.' 'No console to read the password from (use -PasswordStdin).' }
            $script:Pw = Read-Password
        }
        if ($script:Pw) { $script:Pw = $script:Pw.Trim() }
        if (-not $script:Pw) { Say '비밀번호가 비어 있습니다.' 'The password is empty.'; continue }
        Say '비밀번호 확인 중...' 'Checking the password...'
        $code = Invoke-Pipe 'verify'
        if ($code -eq 0) { Say '비밀번호가 확인되었습니다.' 'Password accepted.'; return }
        if ($code -eq 3) {
            if ($script:Pw -match '[^\x20-\x7E]') {
                Say '비밀번호가 맞지 않습니다. 한글로 입력된 것 같습니다: 한/영 키를 누른 뒤 다시 입력하세요.' 'Wrong password. Non-English (e.g. Hangul) input detected: switch the keyboard to English and retry.'
            } else {
                Say '비밀번호가 맞지 않습니다. (한/영 · Caps Lock 확인)' 'Wrong password (check the input language and Caps Lock).'
            }
            $script:Pw = $null
            continue
        }
        if ($code -eq 4) { Die '설치 파일이 손상되었습니다. 키트를 다시 받으세요.' 'The payload is damaged; download the kit again.' }
        Die "비밀번호 확인 중 Docker 오류가 났습니다 (코드 $code)." "Docker error while checking the password (exit $code)."
    }
    Clear-Secret
    Die '비밀번호가 맞지 않습니다. 아무것도 설치하지 않았습니다.' 'Wrong password; nothing was installed.'
}

function Ensure-HelperImage {
    if (Test-Image $HelperImage) { return }
    Say "기본 이미지($HelperImage)를 내려받는 중..." "Pulling $HelperImage..."
    $o = Dk pull $HelperImage
    Write-Log $o
    if ($script:DkCode -ne 0) {
        Die "$HelperImage 을(를) 받지 못했습니다. 인터넷·프록시·방화벽을 확인하세요." "Could not pull $HelperImage (network, proxy or firewall)."
    }
}

function Explain-BuildFailure([string]$DecryptLog) {
    if ((Test-Path -LiteralPath $DecryptLog) -and ((Get-Item -LiteralPath $DecryptLog).Length -gt 0)) {
        $d = [System.IO.File]::ReadAllText($DecryptLog)
        if ($d -match 'error') { Die ('설치 파일을 푸는 중 오류가 났습니다: ' + $d.Trim()) "Decryption failed (see $DecryptLog)." }
    }
    $text = ''
    if (Test-Path -LiteralPath $script:BuildLog) { $text = [System.IO.File]::ReadAllText($script:BuildLog) }
    $net = 'failed to resolve|dial tcp|i/o timeout|TLS handshake|x509|certificate|proxyconnect|ECONNRESET|ETIMEDOUT|EAI_AGAIN|network is unreachable|Temporary failure in name resolution|Could not fetch URL|failed to do request|connection refused|npm ERR! network|Read timed out'
    if ($text -match $net) {
        Say '인터넷 연결 문제로 보입니다. 처음 설치할 때는 Docker Hub, npm, PyPI에서 부품을 내려받습니다.' `
            'This looks like a network problem: the first build downloads from Docker Hub, npm and PyPI.'
        Say '- 회사/학교 프록시가 있으면 Docker Desktop → Settings → Resources → Proxies에 설정하세요.' '- Behind a proxy: set it in Docker Desktop -> Settings -> Resources -> Proxies.'
        Say '- 방화벽/보안 프로그램이 docker.io, registry.npmjs.org, pypi.org, files.pythonhosted.org를 막는지 확인하세요.' `
            '- Check that a firewall does not block docker.io, registry.npmjs.org, pypi.org, files.pythonhosted.org.'
        Say "- 'x509/certificate' 오류는 보안 프로그램의 HTTPS 검사 때문입니다 (docs/LOCAL_INSTALL.md 참고)." "- 'x509/certificate' errors come from HTTPS inspection."
    } elseif ($text -match 'no space left on device') {
        Say '디스크 공간이 부족합니다. Docker Desktop → Troubleshoot → Clean/Purge data 또는 불필요한 이미지를 지우세요.' 'Out of disk space.'
    }
    Write-Host ''
    Write-Host '--- build log (last 25 lines) ---'
    if (Test-Path -LiteralPath $script:BuildLog) { Get-Content -LiteralPath $script:BuildLog -Tail 25 | ForEach-Object { Write-Host $_ } }
    Die '이미지 빌드에 실패했습니다.' ("The image build failed (log: " + $script:BuildLog + ")")
}

function Build-Image([string]$Tag) {
    $stamp = Get-Stamp
    $script:BuildLog = Join-Path $script:InstallDir ("logs\build-" + $script:Kit['VERSION'] + "-$stamp.log")
    $decryptLog = Join-Path $script:InstallDir "logs\decrypt-$stamp.log"
    $env:STL_IMAGE = $Tag
    $env:STL_VERSION = $script:Kit['VERSION']
    $env:STL_BUILD_ID = $script:Kit['BUILD_ID']
    $env:STL_DECRYPT_LOG = $decryptLog
    $env:STL_PROGRESS_ARG = ''
    [void](Dk buildx version)
    if ($script:DkCode -eq 0) { $env:STL_PROGRESS_ARG = '--progress=plain' }
    Say "이미지를 만듭니다: $Tag  (처음에는 5–15분 걸립니다. 정상입니다.)" "Building $Tag (the first build takes 5-15 minutes; this is normal)."
    Say ('  자세한 기록 / full log: ' + $script:BuildLog)
    $code = Invoke-Pipe 'build'
    Clear-Secret
    Prune-BuildContext
    if ($code -ne 0) { Explain-BuildFailure $decryptLog }
    if ((Test-Path -LiteralPath $decryptLog) -and ((Get-Item -LiteralPath $decryptLog).Length -eq 0)) {
        Remove-Item -LiteralPath $decryptLog -ErrorAction SilentlyContinue
    }
    Write-Log (Dk tag $Tag ($Repo + ':local'))
    Say "이미지 빌드 완료 ($Tag)." 'Image built.'
}

# BuildKit caches the build context of "docker build -", i.e. the decrypted source archive (the downloaded
# "http url http://buildkit-session/..." record and its unpacked "copy /context /" snapshot). Nothing needs them after
# the build: delete exactly those records (other projects' build cache is not touched). "." stands for the spaces:
# Windows PowerShell 5.1 does not escape embedded quotes in native-command arguments, so the filter must contain
# neither spaces nor quotes.
function Prune-BuildContext {
    [void](Dk buildx version)
    if ($script:DkCode -ne 0) { return }
    Write-Log (Dk buildx prune -f --filter 'description~=^http.url.http://buildkit-session/')
    Write-Log (Dk buildx prune -f --filter 'description~=^copy./context./$')
}

# Run-Container IMAGE PORT -> 0 ok, 2 port busy, 1 other error
function Run-Container([string]$Image, [int]$Port) {
    [void](Dk rm -f $Container)
    $out = Dk run -d --name $Container --restart unless-stopped -p "127.0.0.1:${Port}:8000" `
        -e ('STL_WORKERS=' + $script:S['WORKERS']) -e FORWARDED_ALLOW_IPS=127.0.0.1 -e 'STL_ALLOWED_HOSTS=127.0.0.1,localhost' `
        -v ($Volume + ':/app/server/.cache') `
        --label ($Label + '=1') --label ($Label + '.dir=' + $script:InstallDir) $Image
    $code = $script:DkCode
    Write-Log "docker run $Image on 127.0.0.1:$Port -> $code $out"
    if ($code -eq 0) { return 0 }
    [void](Dk rm -f $Container)
    if ($out -match 'port is already allocated|address already in use|bind|Only one usage of each socket') { return 2 }
    Write-Host $out
    return 1
}

function Wait-Health([string]$Port) {
    $waited = 0
    while ($waited -lt $HealthTimeout) {
        $body = Get-Local "http://127.0.0.1:$Port/api/health"
        if ($body -match '"ok"\s*:\s*true') { return $true }
        $restarts = (Dk container inspect -f '{{.RestartCount}}' $Container).Trim()
        if ((-not (Test-ContainerRunning)) -or ($restarts -ne '0')) {
            Write-Log ('container stopped or crashed: ' + (Dk logs --tail 30 $Container))
            return $false
        }
        if (($waited -gt 0) -and (($waited % 20) -eq 0)) { Say "  시작을 기다리는 중... ($waited s)" '  waiting for the server...' }
        Start-Sleep -Seconds 2
        $waited += 2
    }
    return $false
}

function Start-OnPort([string]$Image) {
    if ($script:Opt.Port) {
        $r = Run-Container $Image ([int]$script:Opt.Port)
        if ($r -eq 0) { $script:S['PORT'] = [string]$script:Opt.Port; return $true }
        if ($r -eq 2) { Die ("포트 " + $script:Opt.Port + " 을(를) 다른 프로그램이 쓰고 있습니다.") ("Port " + $script:Opt.Port + " is in use.") }
        return $false
    }
    $cands = @()
    if ($script:S['PORT']) { $cands += [int]$script:S['PORT'] }
    $cands += ($PortMin..$PortMax)
    foreach ($p in $cands) {
        if (([string]$p -ne [string]$script:S['PORT']) -and (Test-PortInUse $p)) { continue }
        $r = Run-Container $Image $p
        if ($r -eq 0) { $script:S['PORT'] = [string]$p; return $true }
        if ($r -ne 2) { return $false }
    }
    Die "포트 $PortMin–$PortMax 이 모두 사용 중입니다. -Port 로 다른 번호를 지정하세요." "Ports $PortMin-$PortMax are all busy; use -Port."
}

function Prune-Images {
    $tags = Dk images $Repo --format '{{.Tag}}'
    foreach ($t in ($tags -split "`n")) {
        $t = $t.Trim()
        if (-not $t.StartsWith('local-')) { continue }
        $img = $Repo + ':' + $t
        if (($img -eq $script:S['CURRENT_IMAGE']) -or ($img -eq $script:S['PREVIOUS_IMAGE'])) { continue }
        Write-Log ('prune ' + $img + ': ' + (Dk rmi $img))
    }
    Write-Log (Dk image prune -f --filter ('label=' + $Label + '=1'))
}

# ----------------------------------------------------------------------------------------------- files
function Write-Bin {
    $bin = Join-Path $script:InstallDir 'bin'
    New-Item -ItemType Directory -Force -Path $bin, (Join-Path $script:InstallDir 'logs') | Out-Null
    $target = Join-Path $bin 'stl-sim.ps1'
    if ($ScriptPath -ne $target) { Copy-Item -LiteralPath $ScriptPath -Destination $target -Force }
    try { Unblock-File -LiteralPath $target -ErrorAction SilentlyContinue } catch { }
    $wrappers = @{ start = '-Pause'; stop = '-Pause'; status = '-Pause'; open = '-PauseOnError'; update = '-Pause'
                   rollback = '-Pause'; uninstall = '-Pause'; logs = '-Pause' }
    foreach ($c in $wrappers.Keys) {
        $text = "@rem STL Simulator: $c`n" +
                "@powershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0stl-sim.ps1`" $c $($wrappers[$c]) %* & exit /b`n"
        Write-TextFile (Join-Path $bin "$c.bat") $text 'ascii'
    }
}

function Write-VersionFile {
    $lines = @('STL Simulator (local)',
               ('version:     ' + $script:S['CURRENT_VERSION']),
               ('build id:    ' + $script:S['CURRENT_BUILD']),
               ('commit date: ' + $script:Kit['COMMIT_DATE']),
               ('kit built:   ' + $script:Kit['BUILT']),
               ('installed:   ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')),
               ('image:       ' + $script:S['CURRENT_IMAGE']),
               ('previous:    ' + $(if ($script:S['PREVIOUS_IMAGE']) { $script:S['PREVIOUS_IMAGE'] } else { 'none' })),
               ('address:     http://127.0.0.1:' + $script:S['PORT']))
    Write-TextFile (Join-Path $script:InstallDir 'version.txt') (($lines -join "`n") + "`n") 'bom'
}

function Write-Readme {
    $p = $script:S['PORT']
    $t = @"
STL Simulator 사용법 (로컬 설치, 버전 $($script:S['CURRENT_VERSION']))
==============================================

주소: http://127.0.0.1:$p   (브라우저 주소창에 입력)
설치 폴더: $($script:InstallDir)

열기:        바탕화면 또는 시작 메뉴의 "STL Simulator" 아이콘 (또는 bin\open.bat)
시작 / 중지: bin\start.bat  /  bin\stop.bat
상태 확인:   bin\status.bat
업데이트:    새 설치 키트를 받으면 그 키트의 install-windows.bat 을 다시 실행 (설정·포트 유지)
되돌리기:    bin\rollback.bat  (바로 전 버전으로)
제거:        bin\uninstall.bat
기록(로그):  logs 폴더

- Docker Desktop이 켜져 있어야 합니다. 시뮬레이터는 Docker Desktop이 시작될 때 자동으로 함께 시작됩니다
  (중지 명령으로 멈춘 경우 제외). 쓰지 않을 때 메모리를 아끼려면 stop.bat 을 실행하세요.
- 이 컴퓨터에서만 접속됩니다 (127.0.0.1). 다른 사람과 주소를 공유할 수 없습니다.
- 문제가 있으면 logs 폴더를 담당자에게 보내 주세요 (비밀번호는 기록되지 않습니다).

English: open http://127.0.0.1:$p ; bin\ has start, stop, open, status, rollback, uninstall (.bat);
to update, run install-windows.bat of a new kit.
"@
    Write-TextFile (Join-Path $script:InstallDir 'README-사용법.txt') $t 'bom'
}

function Get-ShortcutPaths {
    $paths = @()
    try { $paths += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'STL Simulator.lnk') } catch { }
    try { $paths += (Join-Path ([Environment]::GetFolderPath('Programs')) 'STL Simulator.lnk') } catch { }
    return $paths
}

function Make-Shortcuts {
    if ($script:Opt.NoShortcut) { return }
    try {
        $ws = New-Object -ComObject WScript.Shell
        foreach ($p in (Get-ShortcutPaths)) {
            $lnk = $ws.CreateShortcut($p)
            $lnk.TargetPath = Join-Path $script:InstallDir 'bin\open.bat'
            $lnk.WorkingDirectory = Join-Path $script:InstallDir 'bin'
            $lnk.Description = 'STL Simulator (local)'
            $lnk.IconLocation = (Join-Path $env:SystemRoot 'System32\shell32.dll') + ',13'
            $lnk.Save()
        }
        Say '바탕화면과 시작 메뉴에 "STL Simulator" 아이콘을 만들었습니다.' 'Created "STL Simulator" shortcuts (desktop, Start menu).'
    } catch {
        Write-Log ('shortcut: ' + $_)
        Say ('아이콘을 만들지 못했습니다 (사용에는 문제 없음). 열기: ' + (Join-Path $script:InstallDir 'bin\open.bat')) 'Could not create shortcuts (not a problem).'
    }
}

function Remove-Shortcuts {
    foreach ($p in (Get-ShortcutPaths)) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
}

# ----------------------------------------------------------------------------------------------- commands
function Read-Kit {
    $script:Kit = Get-KeyValueFile (Join-Path $ScriptDir 'kit-info.txt')
    $v = [string]$script:Kit['VERSION']
    if ((-not $v) -or ($v -notmatch '^[0-9A-Za-z._-]+$')) { Die 'kit-info.txt가 올바르지 않습니다.' 'Bad kit-info.txt.' }
    $payload = Join-Path $ScriptDir ([string]$script:Kit['PAYLOAD'])
    if (-not (Test-Path -LiteralPath $payload)) { Die '설치 파일이 없습니다. ZIP 압축을 전부 풀었는지 확인하세요.' 'The payload file is missing; extract the whole zip.' }
    $want = [string]$script:Kit['PAYLOAD_SHA256']
    if ($want -and ((Get-Sha256 $payload) -ne $want)) {
        Die '설치 파일이 손상되었습니다 (다운로드가 덜 되었을 수 있음). 키트를 다시 받으세요.' 'The payload is damaged or incomplete; download the kit again.'
    }
    $script:HelperB64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes((Join-Path $ScriptDir 'stl_payload.py')))
    # cmd.exe cannot use a network path (\\server\share\...) as its current folder: run the pipe from a local copy.
    $script:PipeDir = $ScriptDir
    if ($ScriptDir.StartsWith('\\')) {
        $script:TmpKit = Join-Path ([System.IO.Path]::GetTempPath()) ('stl-kit-' + (Get-Stamp))
        Say '네트워크 위치에서 실행 중이라 설치 파일을 임시 폴더로 복사합니다...' 'Running from a network path; copying the kit files to a local temporary folder...'
        Copy-Item -LiteralPath $ScriptDir -Destination $script:TmpKit -Recurse -Force
        $script:PipeDir = $script:TmpKit
    }
}

function Cmd-Install {
    if (-not $KitMode) { Die '설치는 설치 키트의 install-windows.bat 으로 실행하세요.' 'Run install from the installer kit.' }
    Read-Kit
    $script:TmpLog = Join-Path ([System.IO.Path]::GetTempPath()) ('stl-install-' + (Get-Stamp) + '.log')
    $script:LogFile = $script:TmpLog
    $ver = $script:Kit['VERSION']
    Write-Host ''
    Say "STL Simulator 설치 (버전 $ver)" "STL Simulator installer (version $ver)"
    Say '진행 순서: Docker 확인 → 비밀번호 확인 → 시뮬레이터 이미지 만들기(처음 5–15분) → 실행 → 브라우저 열기' `
        'Steps: check Docker -> check the password -> build the simulator image (first time 5-15 min) -> start -> open the browser.'

    Step '[1/6] Docker 확인' 'Checking Docker'
    if (-not (Find-Docker)) { Offer-DockerInstall }
    Ensure-Engine
    Check-Platform

    Step '[2/6] 설치 위치' 'Install folder'
    Resolve-InstallDir
    $existing = $false
    if ((-not (Test-OurDir $script:InstallDir)) -and (-not $script:Opt.InstallDir) -and (-not $script:Opt.Yes) -and (Test-Interactive)) {
        Write-Host ('설치 폴더 [' + $script:InstallDir + ']')
        $ans = Read-Host '  Install folder (Enter = default)'
        if ($ans -and $ans.Trim()) { $script:Opt.InstallDir = $ans; Resolve-InstallDir }
    }
    Load-State
    if (Test-OurDir $script:InstallDir) {
        if ($script:S['CURRENT_IMAGE']) {
            $existing = $true
            Say ("기존 설치를 찾았습니다 (버전 " + $script:S['CURRENT_VERSION'] + ") → 업데이트/복구합니다: " + $script:InstallDir) `
                ("Existing install found (version " + $script:S['CURRENT_VERSION'] + "); updating/repairing.")
        } else {
            Say ('설치 폴더: ' + $script:InstallDir + ' (이전에 끝나지 않은 설치를 이어서 합니다)') 'Install folder (resuming an unfinished install)'
        }
    } elseif ((Test-Path -LiteralPath $script:InstallDir) -and
              (@(Get-ChildItem -LiteralPath $script:InstallDir -Force -ErrorAction SilentlyContinue).Count -gt 0)) {
        Die ('폴더가 비어 있지 않습니다: ' + $script:InstallDir + ' (다른 폴더를 지정하세요)') 'The folder is not empty; choose another one (-InstallDir).'
    } else {
        Say ('설치 폴더: ' + $script:InstallDir) 'Install folder'
    }
    if ($existing) { Check-Disk 3 } else { Check-Disk 5 }

    Step '[3/6] 비밀번호 확인' 'Password'
    Ensure-HelperImage
    Get-Password

    New-Item -ItemType Directory -Force -Path (Join-Path $script:InstallDir 'bin'), (Join-Path $script:InstallDir 'logs') | Out-Null
    Write-TextFile (Join-Path $script:InstallDir $Marker) "STL Simulator install folder (used by the installer and uninstaller)`n" 'bom'
    $script:LogFile = Join-Path $script:InstallDir ('logs\install-' + (Get-Stamp) + '.log')
    if (Test-Path -LiteralPath $script:TmpLog) {
        try { [System.IO.File]::AppendAllText($script:LogFile, [System.IO.File]::ReadAllText($script:TmpLog)) } catch { }
        Remove-TmpLog
    }
    $script:TmpLog = $null

    Step '[4/6] 시뮬레이터 이미지 만들기' 'Building the simulator image'
    $tag = $Repo + ':local-' + $ver
    $oldImage = [string]$script:S['CURRENT_IMAGE']
    $oldVersion = [string]$script:S['CURRENT_VERSION']
    Build-Image $tag

    Step '[5/6] 실행' 'Starting'
    $script:S['WORKERS'] = [string]$script:Workers
    $ok = Start-OnPort $tag
    if ($ok) { $ok = Wait-Health $script:S['PORT'] }
    if (-not $ok) {
        Write-Log (Dk logs --tail 40 $Container)
        if ($oldImage -and ($oldImage -ne $tag) -and (Test-Image $oldImage)) {
            Warn "새 버전이 시작되지 않아 이전 버전($oldVersion)으로 되돌립니다." "The new version did not start; rolling back to $oldVersion."
            if (((Run-Container $oldImage ([int]$script:S['PORT'])) -eq 0) -and (Wait-Health $script:S['PORT'])) {
                [void](Dk tag $oldImage ($Repo + ':local'))
                Write-Log (Dk rmi $tag)
                Die '업데이트에 실패해 이전 버전으로 되돌렸습니다. logs 폴더를 담당자에게 보내 주세요.' 'The update failed; the previous version is running again.'
            }
        }
        [void](Dk rm -f $Container)
        Die '시뮬레이터가 시작되지 않았습니다. logs 폴더를 담당자에게 보내 주세요.' 'The simulator did not start (its output is in the install log in the logs folder).'
    }
    if ($oldImage -and ($oldImage -ne $tag)) { $script:S['PREVIOUS_IMAGE'] = $oldImage; $script:S['PREVIOUS_VERSION'] = $oldVersion }
    $script:S['CURRENT_IMAGE'] = $tag
    $script:S['CURRENT_VERSION'] = $ver
    $script:S['CURRENT_BUILD'] = $script:Kit['BUILD_ID']
    $script:S['INSTALLED'] = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    Save-State
    $url = 'http://127.0.0.1:' + $script:S['PORT'] + '/'
    Say "시뮬레이터가 실행 중입니다: $url" "The simulator is running: $url"

    Step '[6/6] 마무리' 'Finishing'
    Write-Bin
    Write-VersionFile
    Write-Readme
    Make-Shortcuts
    Prune-Images
    if (-not $script:Opt.NoBrowser) { if (-not (Open-Url $url)) { Say "브라우저에서 $url 을 여세요." "Open $url in your browser." } }
    $b = Join-Path $script:InstallDir 'bin'
    Write-Host ''
    Say "설치 완료! 주소: $url" "Done! Address: $url"
    Say "  시작/중지/상태: $b\start.bat, stop.bat, status.bat" '  start / stop / status: bin\*.bat'
    Say '  업데이트: 새 키트의 install-windows.bat 실행   되돌리기: bin\rollback.bat   제거: bin\uninstall.bat' `
        "  update: run a new kit's install-windows.bat; rollback: bin\rollback.bat; uninstall: bin\uninstall.bat"
    Say ('  사용법: ' + (Join-Path $script:InstallDir 'README-사용법.txt')) '  how-to: README-사용법.txt'
    Pause-End
}

function Require-Install {
    Resolve-InstallDir
    if (-not (Test-OurDir $script:InstallDir)) { Die ('설치된 STL Simulator를 찾지 못했습니다 (' + $script:InstallDir + ').') 'No installation found.' }
    Load-State
    New-Item -ItemType Directory -Force -Path (Join-Path $script:InstallDir 'logs') | Out-Null
    $script:LogFile = Join-Path $script:InstallDir 'logs\control.log'
}

function Cmd-Start {
    Require-Install
    if (-not (Find-Docker)) { Die 'Docker Desktop을 찾지 못했습니다.' 'Docker CLI not found.' }
    Ensure-Engine
    if (Test-ContainerRunning) { }
    elseif (Test-ContainerExists) {
        Write-Log (Dk start $Container)
        if ($script:DkCode -ne 0) { Die '컨테이너를 시작하지 못했습니다.' 'Could not start the container.' }
    } else {
        $img = [string]$script:S['CURRENT_IMAGE']
        if (-not $img) { $img = $Repo + ':local' }
        if (-not (Test-Image $img)) { Die '시뮬레이터 이미지가 없습니다. 설치 키트로 다시 설치하세요.' 'The image is missing; run the installer kit again.' }
        if (-not $script:S['WORKERS']) { $script:S['WORKERS'] = '1' }
        if (-not (Start-OnPort $img)) { Die '컨테이너를 시작하지 못했습니다.' 'Could not start the container.' }
        Save-State
    }
    Say '시작하는 중...' 'Starting...'
    if (-not (Wait-Health $script:S['PORT'])) { Die "시뮬레이터가 응답하지 않습니다. 'docker logs $Container'를 확인하세요." 'The simulator does not respond.' }
    Say ('실행 중: http://127.0.0.1:' + $script:S['PORT']) ('Running: http://127.0.0.1:' + $script:S['PORT'])
}

function Cmd-Stop {
    Require-Install
    if ((-not (Find-Docker)) -or (-not (Test-Engine))) { Say 'Docker가 꺼져 있습니다 (이미 중지됨).' 'Docker is not running (already stopped).'; return }
    Write-Log (Dk stop $Container)
    Say '중지했습니다. 다시 시작: bin\start.bat' 'Stopped. Start again with bin\start.bat.'
}

function Cmd-Status {
    Require-Install
    Say ('설치 폴더: ' + $script:InstallDir) 'Install folder'
    $prev = [string]$script:S['PREVIOUS_VERSION']
    if (-not $prev) { $prev = '없음' }
    Say ('버전: ' + $script:S['CURRENT_VERSION'] + '  (이전: ' + $prev + ')') 'Version (previous)'
    if ((-not (Find-Docker)) -or (-not (Test-Engine))) { Say 'Docker: 꺼짐' 'Docker is not running'; return }
    $st = (Dk container inspect -f '{{.State.Status}}' $Container).Trim()
    if ($script:DkCode -ne 0) { $st = '없음' }
    Say ('컨테이너: ' + $st + '   주소: http://127.0.0.1:' + $script:S['PORT']) ('Container: ' + $st)
    if ($st -eq 'running') {
        if ((Get-Local ('http://127.0.0.1:' + $script:S['PORT'] + '/api/health')) -match '"ok"\s*:\s*true') { Say '상태: 정상' 'Health: ok' }
        else { Say '상태: 응답 없음 (시작 중일 수 있음)' 'Health: no answer (maybe still starting)' }
    }
}

function Cmd-Rollback {
    Require-Install
    if (-not (Find-Docker)) { Die 'Docker Desktop을 찾지 못했습니다.' 'Docker CLI not found.' }
    Ensure-Engine
    $prev = [string]$script:S['PREVIOUS_IMAGE']
    if (-not (Test-Image $prev)) { Die '되돌릴 이전 버전이 없습니다.' 'No previous version to roll back to.' }
    Say ('이전 버전(' + $script:S['PREVIOUS_VERSION'] + ')으로 되돌립니다...') ('Rolling back to ' + $script:S['PREVIOUS_VERSION'] + '...')
    if (-not $script:S['WORKERS']) { $script:S['WORKERS'] = '1' }
    if ((Run-Container $prev ([int]$script:S['PORT'])) -ne 0) { Die '이전 버전을 시작하지 못했습니다.' 'Could not start the previous version.' }
    if (-not (Wait-Health $script:S['PORT'])) { Die '이전 버전이 응답하지 않습니다.' 'The previous version does not respond.' }
    [void](Dk tag $prev ($Repo + ':local'))
    $ci = $script:S['CURRENT_IMAGE']; $cv = $script:S['CURRENT_VERSION']
    $script:S['CURRENT_IMAGE'] = $prev; $script:S['CURRENT_VERSION'] = $script:S['PREVIOUS_VERSION']
    $script:S['PREVIOUS_IMAGE'] = $ci; $script:S['PREVIOUS_VERSION'] = $cv
    $labels = Dk image inspect -f '{{json .Config.Labels}}' $prev
    try { $script:S['CURRENT_BUILD'] = [string](($labels | ConvertFrom-Json).PSObject.Properties[$Label + '.build'].Value) } catch { }
    Save-State
    Write-VersionFile
    Write-Readme
    Say ('되돌렸습니다: 버전 ' + $script:S['CURRENT_VERSION'] + ', http://127.0.0.1:' + $script:S['PORT'] + ' (한 번 더 실행하면 버전 ' + $script:S['PREVIOUS_VERSION'] + ' 으로 돌아갑니다)') `
        ('Rolled back to ' + $script:S['CURRENT_VERSION'] + ' (run rollback again to return to ' + $script:S['PREVIOUS_VERSION'] + ').')
}

function Cmd-Update {
    if ($script:Opt.Rollback) { Cmd-Rollback; return }
    Require-Install
    $kit = $script:Opt.Kit
    if (-not $kit) {
        Say '업데이트하려면 새 설치 키트(ZIP)의 압축을 풀고 그 안의 install-windows.bat 을 실행하세요.' 'To update, unzip the new kit and run its install-windows.bat.'
        Say '이전 버전으로 되돌리기: bin\rollback.bat' 'Previous version: bin\rollback.bat'
        return
    }
    foreach ($s in @((Join-Path $kit 'installer-files\install-windows.ps1'), (Join-Path $kit 'install-windows.ps1'))) {
        if (Test-Path -LiteralPath $s) {
            $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $s, 'install', '-InstallDir', $script:InstallDir)
            if ($script:Opt.Yes) { $a += '-Yes' }
            if ($script:Opt.Pause) { $a += '-Pause' }
            if ($script:Opt.NoBrowser) { $a += '-NoBrowser' }
            if ($script:Opt.NoShortcut) { $a += '-NoShortcut' }
            if ($script:Opt.Port) { $a += @('-Port', [string]$script:Opt.Port) }
            & powershell.exe @a
            exit $LASTEXITCODE
        }
    }
    Die ('키트 폴더가 아닙니다: ' + $kit + ' (ZIP이면 먼저 압축을 푸세요)') 'Not a kit folder (unzip it first).'
}

function Cmd-Uninstall {
    $bcKept = $false
    Require-Install
    if (-not (Ask ('STL Simulator를 제거할까요? (' + $script:InstallDir + ')') 'Uninstall the STL Simulator?' $false)) { Say '취소했습니다.' 'Cancelled.'; return }
    if ((Find-Docker) -and (Test-Engine)) {
        [void](Dk rm -f $Container)
        foreach ($img in ((Dk images $Repo --format '{{.Repository}}:{{.Tag}}') -split "`n")) {
            $img = $img.Trim()
            if ((-not $img) -or $img.EndsWith(':<none>')) { continue }
            [void](Dk rmi -f $img)
            if ($script:DkCode -eq 0) { Say ('  이미지 삭제: ' + $img) ('  removed ' + $img) }
        }
        [void](Dk image prune -f --filter ('label=' + $Label + '=1'))
        $rmCache = $true
        if ($script:Opt.Cache -eq 'keep') { $rmCache = $false }
        elseif ($script:Opt.Cache -ne 'remove') { $rmCache = Ask "계산 결과 캐시(Docker 볼륨 $Volume)도 지울까요?" 'Also delete the result cache volume?' $true }
        if ($rmCache) { [void](Dk volume rm $Volume); if ($script:DkCode -eq 0) { Say ('  캐시 삭제: ' + $Volume) ('  removed volume ' + $Volume) } }
        Prune-BuildContext
        $rmBc = $true
        if ($script:Opt.BuildCache -eq 'keep') { $rmBc = $false }
        elseif ($script:Opt.BuildCache -ne 'remove') {
            $rmBc = Ask 'Docker 빌드 캐시도 지울까요? (시뮬레이터 프로그램의 사본이 들어 있습니다. 다른 Docker 프로젝트의 빌드 캐시도 함께 지워집니다)' `
                "Also delete Docker's build cache? It holds a copy of the program (other projects' build cache goes too)." $true
        }
        if ($rmBc) {
            [void](Dk buildx version)
            if ($script:DkCode -eq 0) { [void](Dk buildx prune -af) } else { [void](Dk builder prune -af) }
            if ($script:DkCode -eq 0) { Say '  빌드 캐시 삭제' "  removed Docker's build cache" }
        } else { $bcKept = $true }
    } else {
        Warn "Docker가 꺼져 있어 컨테이너/이미지는 지우지 못했습니다. Docker를 켠 뒤: docker rm -f $Container" 'Docker is not running; remove the container and images later.'
    }
    Remove-Shortcuts
    $dir = $script:InstallDir
    $script:LogFile = $null
    Say ('제거했습니다. 폴더를 잠시 후 삭제합니다: ' + $dir) ('Uninstalled; the folder is deleted in a few seconds: ' + $dir)
    Say '(기본 이미지 python:3.11-slim, node:22-slim 은 남겨 둡니다. 프로그램은 들어 있지 않습니다)' `
        '(The base images python:3.11-slim and node:22-slim are kept; they hold no part of the program.)'
    if ($bcKept) {
        Say 'Docker 빌드 캐시에는 아직 프로그램 사본이 남아 있습니다. 지우려면: docker buildx prune -af (또는 Docker Desktop → Builds)' `
            "Docker's build cache still holds a copy of the program; delete it with: docker buildx prune -af"
    }
    Pause-End
    # This script and the .bat that started it live in the folder: delete it from a detached cmd after they exit.
    [Environment]::CurrentDirectory = [System.IO.Path]::GetTempPath()
    Set-Location -LiteralPath ([System.IO.Path]::GetTempPath())
    if (Test-OurDir $dir) {
        $cmdArgs = '/d /s /c "ping -n 4 127.0.0.1 >nul & rmdir /s /q "' + $dir + '""'
        Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdArgs -WindowStyle Hidden -WorkingDirectory ([System.IO.Path]::GetTempPath())
    }
}

function Show-Usage {
    Get-Content -LiteralPath $ScriptPath -TotalCount 20 | Select-Object -Skip 1 | ForEach-Object { Write-Host $_ }
}

function Parse-Arguments([object[]]$List) {
    $i = 0
    $cmdSet = $false
    while ($i -lt $List.Count) {
        $a = [string]$List[$i]
        $val = $null
        if ($a -match '^(--?|/)([A-Za-z-]+)=(.*)$') { $key = $Matches[2]; $val = $Matches[3] }
        elseif ($a -match '^(--?|/)([A-Za-z-]+)$') { $key = $Matches[2] }
        else { $key = $null }
        if ($key) {
            $k = $key.ToLowerInvariant().Replace('-', '')
            switch ($k) {
                'yes' { $script:Opt.Yes = $true }
                'y' { $script:Opt.Yes = $true }
                'port' { if ($null -eq $val) { $i++; $val = [string]$List[$i] }; $script:Opt.Port = $val }
                'installdir' { if ($null -eq $val) { $i++; $val = [string]$List[$i] }; $script:Opt.InstallDir = $val }
                'kit' { if ($null -eq $val) { $i++; $val = [string]$List[$i] }; $script:Opt.Kit = $val }
                'passwordstdin' { $script:Opt.PasswordStdin = $true }
                'nobrowser' { $script:Opt.NoBrowser = $true }
                'noshortcut' { $script:Opt.NoShortcut = $true }
                'removecache' { $script:Opt.Cache = 'remove' }
                'keepcache' { $script:Opt.Cache = 'keep' }
                'removebuildcache' { $script:Opt.BuildCache = 'remove' }
                'keepbuildcache' { $script:Opt.BuildCache = 'keep' }
                'rollback' { $script:Opt.Rollback = $true }
                'pause' { $script:Opt.Pause = $true }
                'pauseonerror' { $script:Opt.PauseOnError = $true }
                'help' { $script:Command = 'help'; $cmdSet = $true }
                'h' { $script:Command = 'help'; $cmdSet = $true }
                default { Die ('알 수 없는 옵션: ' + $a) ('Unknown option: ' + $a) }
            }
        } elseif ((-not $cmdSet) -and ($Commands -contains $a.ToLowerInvariant())) {
            $script:Command = $a.ToLowerInvariant(); $cmdSet = $true
        } elseif ($a) {
            $script:Opt.Kit = $a
        }
        $i++
    }
    if ($script:Opt.Port -and ($script:Opt.Port -notmatch '^\d{1,5}$')) { Die '-Port 는 숫자여야 합니다.' '-Port must be a number.' }
}

function Main {
    Parse-Arguments $ScriptArgs
    try {
        switch ($script:Command) {
            'install' { Cmd-Install }
            'start' { Cmd-Start; Pause-End }
            'stop' { Cmd-Stop; Pause-End }
            'restart' { Cmd-Stop; Cmd-Start; Pause-End }
            'open' {
                Cmd-Start
                $url = 'http://127.0.0.1:' + $script:S['PORT'] + '/'
                if (-not (Open-Url $url)) { Say "브라우저에서 $url 을 여세요." "Open $url in your browser." }
            }
            'status' { Cmd-Status; Pause-End }
            'logs' { Require-Install; & docker logs --tail 200 $Container; Pause-End }
            'rollback' { Cmd-Rollback; Pause-End }
            'update' { Cmd-Update; Pause-End }
            'uninstall' { Cmd-Uninstall }
            default { Show-Usage }
        }
    } finally {
        Clear-Secret
        Remove-TmpLog
        Remove-TmpKit
    }
}

if ($env:STL_PS_NO_MAIN -ne '1') { Main }
