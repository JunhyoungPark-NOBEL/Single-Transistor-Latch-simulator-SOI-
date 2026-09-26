# Windows PowerShell 5.1 compatible. The installer is always authenticated before execution.
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$LauncherArguments = @())

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$script:StartupLog = $null

function Write-StudioLine {
    param([string]$Message)
    Write-Host $Message
    if ($script:StartupLog) {
        Add-Content -LiteralPath $script:StartupLog -Value $Message -Encoding UTF8
    }
}

function ConvertTo-WindowsArgument {
    param([AllowEmptyString()][string]$Value)
    # CommandLineToArgvW/CRT quoting, also for paths ending in a backslash.
    $result = New-Object System.Text.StringBuilder
    [void]$result.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$result.Append(('\' * (2 * $slashes + 1)))
        } else {
            [void]$result.Append(('\' * $slashes))
        }
        [void]$result.Append($character)
        $slashes = 0
    }
    [void]$result.Append(('\' * (2 * $slashes)))
    [void]$result.Append('"')
    return $result.ToString()
}

function Invoke-CapturedProcess {
    param([string]$Executable, [string[]]$Arguments, [int]$TimeoutMilliseconds = 10000)
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $Executable
    $start.Arguments = (($Arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' ')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            $process.Kill()
            $process.WaitForExit()
            return $null
        }
        return [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $stdout.Result; Error = $stderr.Result }
    } catch {
        return $null
    } finally {
        $process.Dispose()
    }
}

function Test-StudioPython {
    param([string]$Executable)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    if ($Executable -match '[\\/]Microsoft[\\/]WindowsApps[\\/]') { return $null }
    $probe = @'
import json, platform, struct, sys, sysconfig, venv, ensurepip
ok = (sys.implementation.name == 'cpython' and (3, 11) <= sys.version_info[:2] <= (3, 14) and sys.version_info.releaselevel == 'final' and struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64') and not sysconfig.get_config_var('Py_GIL_DISABLED'))
print(json.dumps({'ok': ok, 'executable': sys.executable, 'version': platform.python_version()}))
sys.exit(0 if ok else 1)
'@
    $result = Invoke-CapturedProcess -Executable $Executable -Arguments @('-I', '-c', $probe)
    if ($null -eq $result -or $result.ExitCode -ne 0) { return $null }
    try {
        $info = $result.Output.Trim() | ConvertFrom-Json
        if ($info.ok -and $info.executable -and $info.version) { return $info }
    } catch { }
    return $null
}

function Get-PythonCandidates {
    param([string]$AppRoot, [string]$RuntimeRoot, [object]$Manifest)
    # Prefer an already prepared project or app-private runtime. Never invoke Store aliases.
    Join-Path $AppRoot '.venv\Scripts\python.exe'
    Join-Path (Join-Path $RuntimeRoot $Manifest.targetDirectory) 'python.exe'
    $launcher = Get-Command 'py.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($launcher -and $launcher.Source -notmatch '[\\/]Microsoft[\\/]WindowsApps[\\/]') {
        # Listing installed versions cannot trigger an automatic Python-manager installation.
        $listed = Invoke-CapturedProcess -Executable $launcher.Source -Arguments @('-0p')
        if ($null -ne $listed -and $listed.ExitCode -eq 0) {
            foreach ($line in ($listed.Output -split '\r?\n')) {
                if ($line -match '([A-Za-z]:\\.+?python\.exe)\s*$') { $Matches[1] }
            }
        }
    }
    foreach ($base in @('HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore', 'HKLM:\Software\WOW6432Node\Python\PythonCore')) {
        if (-not (Test-Path -LiteralPath $base)) { continue }
        foreach ($version in (Get-ChildItem -LiteralPath $base -ErrorAction SilentlyContinue)) {
            $key = Get-Item -LiteralPath ($version.PSPath + '\InstallPath') -ErrorAction SilentlyContinue
            if ($key) {
                $executable = $key.GetValue('ExecutablePath')
                if ($executable) { [string]$executable }
                $directory = $key.GetValue('')
                if ($directory) { Join-Path ([string]$directory) 'python.exe' }
            }
        }
    }
    foreach ($name in @('python.exe', 'python3.exe')) {
        foreach ($command in (Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)) {
            if ($command.Source -notmatch '[\\/]Microsoft[\\/]WindowsApps[\\/]') { $command.Source }
        }
    }
}

function Find-StudioPython {
    param([string]$AppRoot, [string]$RuntimeRoot, [object]$Manifest)
    foreach ($candidate in (Get-PythonCandidates -AppRoot $AppRoot -RuntimeRoot $RuntimeRoot -Manifest $Manifest | Select-Object -Unique)) {
        $info = Test-StudioPython -Executable $candidate
        if ($null -ne $info) { return $info }
    }
    return $null
}

function Get-RuntimeManifest {
    param([string]$AppRoot)
    $manifest = Get-Content -LiteralPath (Join-Path $AppRoot 'scripts\python-runtime.json') -Raw | ConvertFrom-Json
    $uri = [uri]$manifest.url
    if ($uri.Scheme -ne 'https' -or $uri.Host -ne 'www.python.org' -or $uri.UserInfo -or $uri.Query -or $uri.Fragment -or
        $manifest.sha256 -notmatch '^[0-9a-f]{64}$' -or $manifest.architecture -ne 'amd64' -or
        $manifest.filename -notmatch '^python-3\.13\.\d+-amd64\.exe$' -or $manifest.targetDirectory -ne 'Python313' -or
        $manifest.size -le 0 -or $uri.AbsolutePath -ne ('/ftp/python/' + $manifest.version + '/' + $manifest.filename)) {
        throw 'Invalid bundled Python runtime manifest. Extract a fresh copy of STL simulator.'
    }
    return $manifest
}

function Assert-TrustedInstaller {
    param([string]$Path, [object]$Manifest)
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Manifest.size) {
        throw "Python installer size does not match the official release. Installation stopped: $Path"
    }
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($hash -ine $Manifest.sha256) {
        throw "Python installer SHA-256 verification failed. Installation stopped: $Path"
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $Path
    if ($signature.Status -ne 'Valid' -or $null -eq $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O="?Python Software Foundation"?(,|$)') {
        throw "Python installer signature is not a valid Python Software Foundation signature. Installation stopped: $Path"
    }
    # Windows validates timestamped signatures. Do not reject a timestamped signer solely on NotAfter.
}

function Receive-PythonInstaller {
    param([string]$Destination, [object]$Manifest)
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $oldProgress = $ProgressPreference
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri $Manifest.url -OutFile $Destination -MaximumRedirection 0
    } finally { $ProgressPreference = $oldProgress }
}

function Get-TrustedInstaller {
    param([string]$AppRoot, [string]$RuntimeRoot, [object]$Manifest)
    $bundled = Join-Path (Join-Path $AppRoot 'vendor') $Manifest.filename
    if (Test-Path -LiteralPath $bundled -PathType Leaf) {
        Write-StudioLine '[2/4] Checking the bundled official Python installer ...'
        Assert-TrustedInstaller -Path $bundled -Manifest $Manifest
        return $bundled
    }
    $cache = Join-Path $RuntimeRoot 'cache'
    [void](New-Item -ItemType Directory -Path $cache -Force)
    $cached = Join-Path $cache $Manifest.filename
    if (-not (Test-Path -LiteralPath $cached -PathType Leaf)) {
        Write-StudioLine ('[2/4] Downloading official Python ' + $Manifest.version + ' from python.org (about 29 MB) ...')
        $partial = $cached + '.' + [guid]::NewGuid().ToString('N') + '.partial'
        try {
            Receive-PythonInstaller -Destination $partial -Manifest $Manifest
            Assert-TrustedInstaller -Path $partial -Manifest $Manifest
            Move-Item -LiteralPath $partial -Destination $cached
        } finally {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        }
    }
    Assert-TrustedInstaller -Path $cached -Manifest $Manifest
    return $cached
}

function Invoke-PythonInstaller {
    param([string]$Installer, [string]$TargetDirectory, [string]$InstallerLog)
    $arguments = @('/quiet', '/norestart', '/log', $InstallerLog,
        'InstallAllUsers=0', ('TargetDir=' + $TargetDirectory), 'Include_launcher=0', 'InstallLauncherAllUsers=0',
        'PrependPath=0', 'AppendPath=0', 'AssociateFiles=0', 'Shortcuts=0',
        'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Include_pip=1',
        'Include_dev=1', 'Include_exe=1', 'Include_lib=1', 'Include_freethreaded=0', 'CompileAll=0')
    $commandLine = ($arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' '
    $process = Start-Process -FilePath $Installer -ArgumentList $commandLine -Wait -PassThru
    return $process.ExitCode
}

function Assert-PrivateRuntimeInstall {
    param([string]$TargetDirectory)
    # CPython's installer can enter maintenance mode for a registered same-minor release.
    # Never repair or relocate somebody else's broken per-user installation implicitly.
    $registration = 'HKCU:\Software\Python\PythonCore\3.13\InstallPath'
    if (-not (Test-Path -LiteralPath $registration)) { return }
    $key = Get-Item -LiteralPath $registration
    $registeredDirectory = [string]$key.GetValue('')
    if (-not $registeredDirectory) {
        throw 'An incomplete Python 3.13 registration already exists for this account. Repair that Python installation in Windows Settings before starting STL simulator.'
    }
    $registeredFull = [IO.Path]::GetFullPath($registeredDirectory).TrimEnd([char]'\', [char]'/')
    $targetFull = [IO.Path]::GetFullPath($TargetDirectory).TrimEnd([char]'\', [char]'/')
    if ($registeredFull -ine $targetFull) {
        throw "An existing Python 3.13 installation could not be used: $registeredDirectory . Repair it in Windows Settings, then retry. STL simulator will not change this external installation."
    }
}

function Install-StudioPython {
    param([string]$AppRoot, [string]$RuntimeRoot, [object]$Manifest)
    [void](New-Item -ItemType Directory -Path $RuntimeRoot -Force)
    $lock = $null
    try {
        try {
            $lock = [IO.File]::Open((Join-Path $RuntimeRoot 'bootstrap.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        } catch {
            throw 'Another STL simulator Python setup is running. Wait for it to finish, then start again.'
        }
        $existing = Find-StudioPython -AppRoot $AppRoot -RuntimeRoot $RuntimeRoot -Manifest $Manifest
        if ($null -ne $existing) { return $existing }
        $target = Join-Path $RuntimeRoot $Manifest.targetDirectory
        Assert-PrivateRuntimeInstall -TargetDirectory $target
        $installer = Get-TrustedInstaller -AppRoot $AppRoot -RuntimeRoot $RuntimeRoot -Manifest $Manifest
        $installerLog = Join-Path $AppRoot 'studio-python-install.log'
        Write-StudioLine ('[3/4] Installing Python ' + $Manifest.version + ' for this Windows account ...')
        Write-StudioLine ('Location: ' + $target)
        Write-StudioLine 'System PATH, file associations, and administrator settings will not be changed.'
        $code = Invoke-PythonInstaller -Installer $installer -TargetDirectory $target -InstallerLog $installerLog
        if ($code -ne 0 -and $code -ne 3010) { throw "Python installer failed (exit $code). Installer log: $installerLog" }
        if ($code -eq 3010) { Write-StudioLine 'Windows reported that a restart may be needed; checking Python before continuing.' }
        $installed = Test-StudioPython -Executable (Join-Path $target 'python.exe')
        if ($null -eq $installed) { throw "Installed Python could not be started. Installer log: $installerLog" }
        return $installed
    } finally {
        if ($null -ne $lock) { $lock.Dispose() }
    }
}

function Invoke-StudioLauncher {
    param([string]$Python, [string]$AppRoot, [string[]]$Arguments)
    $oldLocation = Get-Location
    $oldErrorPreference = $ErrorActionPreference
    $oldPythonIOEncoding = $env:PYTHONIOENCODING
    $oldPythonUTF8 = $env:PYTHONUTF8
    $oldConsoleEncoding = [Console]::OutputEncoding
    try {
        $env:PYTHONIOENCODING = 'utf-8'
        $env:PYTHONUTF8 = '1'
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
        Set-Location -LiteralPath $AppRoot
        # Windows PowerShell 5.1 wraps native stderr as ErrorRecord; it is ordinary log output here.
        $ErrorActionPreference = 'Continue'
        & $Python -u (Join-Path $AppRoot 'launch.py') @Arguments 2>&1 | ForEach-Object { Write-StudioLine ([string]$_) }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorPreference
        $env:PYTHONIOENCODING = $oldPythonIOEncoding
        $env:PYTHONUTF8 = $oldPythonUTF8
        [Console]::OutputEncoding = $oldConsoleEncoding
        Set-Location -LiteralPath $oldLocation.Path
    }
}

function Invoke-StudioBootstrap {
    param([string]$AppRoot, [string[]]$Arguments = @())
    if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is not available. Run this launcher in a normal Windows user account.' }
    foreach ($required in @('launch.py', 'server\requirements.txt', 'web\dist\index.html')) {
        if (-not (Test-Path -LiteralPath (Join-Path $AppRoot $required) -PathType Leaf)) {
            throw 'Required application files are missing. Extract the ENTIRE STL-simulator.zip before starting.'
        }
    }
    $architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($architecture -ne 'AMD64') { throw 'This package supports 64-bit Intel/AMD Windows. ARM64 and 32-bit Windows are not supported by this installer.' }
    $manifest = Get-RuntimeManifest -AppRoot $AppRoot
    $runtimeRoot = Join-Path $env:LOCALAPPDATA 'BiristorStudio'
    Write-StudioLine '[1/4] Looking for a compatible 64-bit Python (3.11-3.14) ...'
    $python = Find-StudioPython -AppRoot $AppRoot -RuntimeRoot $runtimeRoot -Manifest $manifest
    if ($null -eq $python) {
        if ($Arguments -contains '--check' -or $Arguments -contains '--no-install' -or $Arguments -contains '--use-current-python' -or $Arguments -contains '--help' -or $Arguments -contains '-h') {
            throw 'No compatible Python is installed. This command does not permit installation. Run start-local.bat without these options to set up Python automatically.'
        }
        $python = Install-StudioPython -AppRoot $AppRoot -RuntimeRoot $runtimeRoot -Manifest $manifest
    } else {
        Write-StudioLine ('Reusing Python ' + $python.version + ': ' + $python.executable)
    }
    Write-StudioLine '[4/4] Preparing STL simulator and starting the local server ...'
    Write-StudioLine 'First setup needs internet access for scientific packages and may take several minutes.'
    Write-StudioLine 'The browser opens after the server is ready. Keep this window open; stop with Ctrl+C.'
    return Invoke-StudioLauncher -Python $python.executable -AppRoot $AppRoot -Arguments $Arguments
}

if ($MyInvocation.InvocationName -ne '.') {
    $appRoot = Split-Path -Parent $PSScriptRoot
    $script:StartupLog = Join-Path $appRoot 'studio-startup.log'
    $exitCode = 1
    try {
        Set-Content -LiteralPath $script:StartupLog -Encoding UTF8 -Value ('STL simulator - ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
        $exitCode = Invoke-StudioBootstrap -AppRoot $appRoot -Arguments $LauncherArguments
    } catch {
        $failureMessage = 'STARTUP FAILED: ' + $_.Exception.Message
        try { Write-StudioLine $failureMessage } catch { Write-Host $failureMessage; Write-Host 'The startup log could not be written. Extract the application to a writable folder, such as Documents.' }
    }
    exit $exitCode
}
