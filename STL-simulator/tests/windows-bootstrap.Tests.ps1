# Dependency-free behavioral checks. Run: powershell -NoProfile -File tests\windows-bootstrap.Tests.ps1
# Mocked installer/Authenticode checks can also run under portable pwsh on Linux.
# These checks do NOT prove that the Windows installer or Windows certificate chain executes successfully.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\scripts\bootstrap-windows.ps1')
$script:Passed = 0
$script:Failed = 0
$script:FixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('Biristor bootstrap tests ' + [guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $script:FixtureRoot)
$script:Manifest = Get-RuntimeManifest -AppRoot (Split-Path -Parent $PSScriptRoot)
$script:PythonInfo = [pscustomobject]@{ executable = 'C:\Users\Kim Park\Python\python.exe'; version = '3.13.15'; ok = $true }

function Assert-Equal {
    param($Actual, $Expected)
    if ($Actual -cne $Expected) { throw "Expected [$Expected], got [$Actual]" }
}
function Assert-Throws {
    param([scriptblock]$Body, [string]$Pattern)
    try { & $Body } catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw "Wrong error: $($_.Exception.Message)" }
        return
    }
    throw 'Expected an exception.'
}
function Test-Case {
    param([string]$Name, [scriptblock]$Body)
    try { & $Body; $script:Passed++; Write-Host ('PASS: ' + $Name) }
    catch { $script:Failed++; Write-Host ('FAIL: ' + $Name + ' - ' + $_.Exception.Message) }
}

$originalLocalAppData = $env:LOCALAPPDATA
$originalArchitecture = $env:PROCESSOR_ARCHITECTURE
$originalWowArchitecture = $env:PROCESSOR_ARCHITEW6432
try {
    $env:LOCALAPPDATA = $script:FixtureRoot
    $env:PROCESSOR_ARCHITECTURE = 'AMD64'
    $env:PROCESSOR_ARCHITEW6432 = ''
    $app = Join-Path $script:FixtureRoot 'app with spaces'
    [void](New-Item -ItemType Directory -Path (Join-Path $app 'server') -Force)
    [void](New-Item -ItemType Directory -Path (Join-Path $app 'web\dist') -Force)
    foreach ($file in @('launch.py', 'server\requirements.txt', 'web\dist\index.html')) {
        Set-Content -LiteralPath (Join-Path $app $file) -Value 'test fixture'
    }

    Test-Case 'Windows argument quoting handles spaces, quotes, empty strings, and trailing backslashes' {
        Assert-Equal (ConvertTo-WindowsArgument 'C:\Users\Kim Park\Python') '"C:\Users\Kim Park\Python"'
        Assert-Equal (ConvertTo-WindowsArgument 'C:\Users\Kim Park\') '"C:\Users\Kim Park\\"'
        Assert-Equal (ConvertTo-WindowsArgument '') '""'
        Assert-Equal (ConvertTo-WindowsArgument 'a"b') '"a\"b"'
        Assert-Equal (ConvertTo-WindowsArgument 'a\"b') '"a\\\"b"'
    }
    Test-Case 'compatible existing Python is reused and launcher arguments are preserved' {
        function Get-RuntimeManifest { return $script:Manifest }
        function Find-StudioPython { return $script:PythonInfo }
        function Install-StudioPython { throw 'Unexpected installation' }
        function Invoke-StudioLauncher {
            param($Python, $AppRoot, $Arguments)
            Assert-Equal $Python $script:PythonInfo.executable
            Assert-Equal $AppRoot $app
            Assert-Equal ($Arguments -join '|') '--port|8101|--no-browser'
            return 0
        }
        Assert-Equal (Invoke-StudioBootstrap -AppRoot $app -Arguments @('--port', '8101', '--no-browser')) 0
    }
    Test-Case 'fresh setup installs Python and uses the returned exact interpreter' {
        function Get-RuntimeManifest { return $script:Manifest }
        function Find-StudioPython { return $null }
        function Install-StudioPython { return $script:PythonInfo }
        function Invoke-StudioLauncher {
            param($Python, $AppRoot, $Arguments)
            Assert-Equal $Python $script:PythonInfo.executable
            Assert-Equal ($Arguments -join '|') '--setup-only'
            return 0
        }
        Assert-Equal (Invoke-StudioBootstrap -AppRoot $app -Arguments @('--setup-only')) 0
    }
    foreach ($flag in @('--check', '--no-install', '--use-current-python', '--help')) {
        Test-Case ("$flag refuses mutation when Python is absent") {
            function Get-RuntimeManifest { return $script:Manifest }
            function Find-StudioPython { return $null }
            function Install-StudioPython { throw 'Unexpected installation' }
            function Invoke-StudioLauncher { throw 'Unexpected launcher execution' }
            Assert-Throws { Invoke-StudioBootstrap -AppRoot $app -Arguments @($flag) } 'does not permit installation'
        }
    }
    Test-Case 'failed SHA-256 stops before checking or executing a signature/installer' {
        function Get-Item { return [pscustomobject]@{ Length = $script:Manifest.size } }
        function Get-FileHash { return [pscustomobject]@{ Hash = ('0' * 64) } }
        function Get-AuthenticodeSignature { throw 'Should not reach signature check' }
        Assert-Throws { Assert-TrustedInstaller -Path 'test.exe' -Manifest $script:Manifest } 'SHA-256 verification failed'
    }
    Test-Case 'untrusted signature stops even when the hash matches' {
        function Get-Item { return [pscustomobject]@{ Length = $script:Manifest.size } }
        function Get-FileHash { return [pscustomobject]@{ Hash = $script:Manifest.sha256 } }
        function Get-AuthenticodeSignature { return [pscustomobject]@{ Status = 'NotTrusted'; SignerCertificate = $null } }
        Assert-Throws { Assert-TrustedInstaller -Path 'test.exe' -Manifest $script:Manifest } 'not a valid Python Software Foundation signature'
    }
    Test-Case 'valid signature from a different publisher is rejected' {
        function Get-Item { return [pscustomobject]@{ Length = $script:Manifest.size } }
        function Get-FileHash { return [pscustomobject]@{ Hash = $script:Manifest.sha256 } }
        function Get-AuthenticodeSignature {
            return [pscustomobject]@{ Status = 'Valid'; SignerCertificate = [pscustomobject]@{ Subject = 'CN=Other, O=Other, C=US' } }
        }
        Assert-Throws { Assert-TrustedInstaller -Path 'test.exe' -Manifest $script:Manifest } 'not a valid Python Software Foundation signature'
    }
    Test-Case 'valid PSF timestamped signature is accepted without a separate certificate-expiry test' {
        function Get-Item { return [pscustomobject]@{ Length = $script:Manifest.size } }
        function Get-FileHash { return [pscustomobject]@{ Hash = $script:Manifest.sha256 } }
        function Get-AuthenticodeSignature {
            return [pscustomobject]@{ Status = 'Valid'; SignerCertificate = [pscustomobject]@{ Subject = 'CN=Python Software Foundation, O=Python Software Foundation, C=US'; NotAfter = [datetime]'2026-08-07' } }
        }
        Assert-TrustedInstaller -Path 'test.exe' -Manifest $script:Manifest
    }
    Test-Case 'bundled installer is authenticated without a download' {
        $bundleApp = Join-Path $script:FixtureRoot 'bundled'
        [void](New-Item -ItemType Directory -Path (Join-Path $bundleApp 'vendor') -Force)
        Set-Content -LiteralPath (Join-Path (Join-Path $bundleApp 'vendor') $script:Manifest.filename) -Value 'fixture'
        function Assert-TrustedInstaller { param($Path, $Manifest); Assert-Equal $Path (Join-Path (Join-Path $bundleApp 'vendor') $script:Manifest.filename) }
        function Receive-PythonInstaller { throw 'Unexpected download' }
        Assert-Equal (Get-TrustedInstaller -AppRoot $bundleApp -RuntimeRoot $script:FixtureRoot -Manifest $script:Manifest) (Join-Path (Join-Path $bundleApp 'vendor') $script:Manifest.filename)
    }
    Test-Case 'download failure cleans only its own partial file and never calls installer' {
        $downloadRoot = Join-Path $script:FixtureRoot 'download failure'
        function Receive-PythonInstaller { param($Destination); Set-Content -LiteralPath $Destination -Value 'partial'; throw 'Network disconnected' }
        function Invoke-PythonInstaller { throw 'Unexpected installer execution' }
        Assert-Throws { Get-TrustedInstaller -AppRoot $app -RuntimeRoot $downloadRoot -Manifest $script:Manifest } 'Network disconnected'
        Assert-Equal @(Get-ChildItem -LiteralPath (Join-Path $downloadRoot 'cache') -File).Count 0
    }
    Test-Case 'nonzero installer exit is reported and its lock is released' {
        $runtime = Join-Path $script:FixtureRoot 'failed installation'
        function Find-StudioPython { return $null }
        function Get-TrustedInstaller { return 'verified.exe' }
        function Invoke-PythonInstaller { return 1603 }
        function Test-StudioPython { throw 'Must not probe after failed installer' }
        Assert-Throws { Install-StudioPython -AppRoot $app -RuntimeRoot $runtime -Manifest $script:Manifest } 'exit 1603'
        $handle = [IO.File]::Open((Join-Path $runtime 'bootstrap.lock'), [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        $handle.Dispose()
    }
    Test-Case 'successful installer is probed at its requested path with spaces' {
        $runtime = Join-Path $script:FixtureRoot 'successful installation'
        function Find-StudioPython { return $null }
        function Get-TrustedInstaller { return 'verified.exe' }
        function Invoke-PythonInstaller {
            param($Installer, $TargetDirectory, $InstallerLog)
            Assert-Equal $Installer 'verified.exe'
            Assert-Equal $TargetDirectory (Join-Path $runtime 'Python313')
            Assert-Equal $InstallerLog (Join-Path $app 'studio-python-install.log')
            return 0
        }
        function Test-StudioPython { param($Executable); Assert-Equal $Executable (Join-Path (Join-Path $runtime 'Python313') 'python.exe'); return $script:PythonInfo }
        Assert-Equal (Install-StudioPython -AppRoot $app -RuntimeRoot $runtime -Manifest $script:Manifest).executable $script:PythonInfo.executable
    }
    Test-Case 'concurrent bootstrap lock prevents a second installation' {
        $runtime = Join-Path $script:FixtureRoot 'locked installation'
        [void](New-Item -ItemType Directory -Path $runtime)
        $handle = [IO.File]::Open((Join-Path $runtime 'bootstrap.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
        function Get-TrustedInstaller { throw 'Unexpected installer access' }
        try { Assert-Throws { Install-StudioPython -AppRoot $app -RuntimeRoot $runtime -Manifest $script:Manifest } 'Another Biristor Studio Python setup is running' }
        finally { $handle.Dispose() }
    }
    Test-Case 'external broken Python registration is protected and app-owned retry remains allowed' {
        $target = Join-Path $script:FixtureRoot 'app-private Python313'
        $script:RegisteredPythonDirectory = Join-Path $script:FixtureRoot 'other Python313'
        function Test-Path { return $true }
        function Get-Item {
            $key = [pscustomobject]@{}
            Add-Member -InputObject $key -MemberType ScriptMethod -Name GetValue -Value { param($Name); return $script:RegisteredPythonDirectory }
            return $key
        }
        Assert-Throws { Assert-PrivateRuntimeInstall -TargetDirectory $target } 'will not change this external installation'
        $script:RegisteredPythonDirectory = $target
        Assert-PrivateRuntimeInstall -TargetDirectory $target
    }
    Test-Case 'installer command uses per-user options and quoted target/log paths' {
        function Start-Process {
            param($FilePath, $ArgumentList, [switch]$Wait, [switch]$PassThru)
            Assert-Equal $FilePath 'C:\app with spaces\python.exe'
            foreach ($expected in @('"InstallAllUsers=0"', '"TargetDir=C:\Users\Kim Park\Python313"', '"Include_launcher=0"', '"PrependPath=0"', '"AppendPath=0"', '"AssociateFiles=0"', '"Shortcuts=0"', '"Include_freethreaded=0"', '"C:\app with spaces\install.log"')) {
                if (-not $ArgumentList.Contains($expected)) { throw ('Missing argument: ' + $expected) }
            }
            if (-not $Wait -or -not $PassThru) { throw 'Installer must be waited on.' }
            return [pscustomobject]@{ ExitCode = 0 }
        }
        Assert-Equal (Invoke-PythonInstaller -Installer 'C:\app with spaces\python.exe' -TargetDirectory 'C:\Users\Kim Park\Python313' -InstallerLog 'C:\app with spaces\install.log') 0
    }
    Test-Case 'File invocation preserves double-dash launcher arguments' {
        $bindingScript = Join-Path $script:FixtureRoot 'binding test.ps1'
        Set-Content -LiteralPath $bindingScript -Encoding UTF8 -Value '[CmdletBinding()]param([Parameter(ValueFromRemainingArguments=$true)][string[]]$LauncherArguments=@()); ConvertTo-Json -Compress -InputObject $LauncherArguments'
        $shell = Join-Path $PSHOME 'pwsh'
        if (-not (Test-Path -LiteralPath $shell)) { $shell = Join-Path $PSHOME 'powershell.exe' }
        $result = Invoke-CapturedProcess -Executable $shell -Arguments @('-NoProfile', '-File', $bindingScript, '--port', '8101', '--check', '--host', '127.0.0.1')
        Assert-Equal $result.ExitCode 0
        Assert-Equal (($result.Output | ConvertFrom-Json) -join '|') '--port|8101|--check|--host|127.0.0.1'
    }
    Test-Case 'actual Python pipeline logs Unicode and stderr, restores encoding, and preserves nonzero exit' {
        $pythonCommand = Get-Command 'python3', 'python' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $pythonCommand) { throw 'This subprocess check requires a test-host Python interpreter.' }
        $fixture = Join-Path $script:FixtureRoot 'unicode launcher'
        [void](New-Item -ItemType Directory -Path $fixture)
        Set-Content -LiteralPath (Join-Path $fixture 'launch.py') -Encoding UTF8 -Value 'import sys; print("\ud55c\uae00"); print("stderr-visible", file=sys.stderr); print("|".join(sys.argv[1:])); sys.exit(7)'
        $previousLog = $script:StartupLog
        $previousIO = $env:PYTHONIOENCODING
        try {
            $script:StartupLog = Join-Path $fixture 'startup.log'
            $env:PYTHONIOENCODING = 'cp1252'
            Assert-Equal (Invoke-StudioLauncher -Python $pythonCommand.Source -AppRoot $fixture -Arguments @('--port', '8123', 'path with spaces')) 7
            $log = Get-Content -LiteralPath $script:StartupLog -Raw -Encoding UTF8
            $korean = [string][char]0xd55c + [string][char]0xae00
            if (-not $log.Contains($korean) -or -not $log.Contains('stderr-visible') -or -not $log.Contains('--port|8123|path with spaces')) { throw 'Native stdout/stderr or arguments were not preserved.' }
            Assert-Equal $env:PYTHONIOENCODING 'cp1252'
        } finally {
            $script:StartupLog = $previousLog
            $env:PYTHONIOENCODING = $previousIO
        }
    }
    Test-Case 'ARM64 host refuses x64 automatic installation' {
        $env:PROCESSOR_ARCHITECTURE = 'ARM64'
        try { Assert-Throws { Invoke-StudioBootstrap -AppRoot $app } 'ARM64 and 32-bit Windows are not supported' }
        finally { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
    }
} finally {
    $env:LOCALAPPDATA = $originalLocalAppData
    $env:PROCESSOR_ARCHITECTURE = $originalArchitecture
    $env:PROCESSOR_ARCHITEW6432 = $originalWowArchitecture
    Remove-Item -LiteralPath $script:FixtureRoot -Recurse -Force
}
Write-Host ("Bootstrap behavioral checks: $script:Passed passed, $script:Failed failed.")
if ($script:Failed) { exit 1 }
