<#
.SYNOPSIS
  Opens Chrome so that ONLY that Chrome goes through the FPGate proxy.

.DESCRIPTION
  - Nothing is changed in Windows: no system proxy, no certificate store.
  - Chrome runs with its own profile folder, so the proxy flag is honoured even if
    another Chrome is already open, and cookies/cache stay separate.
  - The FPGate CA is trusted only inside this Chrome instance
    (--ignore-certificate-errors-spki-list with the CA's public-key hash).
    Chrome will show a yellow bar about an unsupported flag: that is expected.

.EXAMPLE
  .\chrome_via_proxy.ps1 -ProxyHost 192.168.100.37
  .\chrome_via_proxy.ps1 -ProxyHost socfpga -Url https://en.wikipedia.org/wiki/Alps
  .\chrome_via_proxy.ps1 -ProxyHost 192.168.100.37 -InstalledCa    # CA already in the Windows store
#>
param(
    [Parameter(Mandatory = $true)][string]$ProxyHost,
    [int]$Port = 8080,
    [string]$CaFile = (Join-Path $PSScriptRoot "..\certs\ca.crt"),
    [string]$Url = "http://example.com/",
    [string]$Profile = (Join-Path $env:LOCALAPPDATA "FPGate-Chrome"),
    [switch]$InstalledCa
)

$ErrorActionPreference = "Stop"

# --- DER walker: extracts the SubjectPublicKeyInfo of a certificate ------------
function Read-Der([byte[]]$b, [int]$pos) {
    $len = $b[$pos + 1]; $hdr = 2
    if ($len -ge 0x80) {
        $n = $len -band 0x7f; $len = 0
        for ($i = 0; $i -lt $n; $i++) { $len = ($len -shl 8) -bor $b[$pos + 2 + $i] }
        $hdr = 2 + $n
    }
    [pscustomobject]@{ Tag = $b[$pos]; Hdr = $hdr; Len = $len; Pos = $pos }
}

function Get-SpkiHash([string]$path) {
    $pem = (Get-Content $path -Raw) -replace "-----[A-Z ]+-----", "" -replace "\s", ""
    $der = [Convert]::FromBase64String($pem)
    $cert = Read-Der $der 0
    $tbs = Read-Der $der ($cert.Pos + $cert.Hdr)
    $pos = $tbs.Pos + $tbs.Hdr
    $first = Read-Der $der $pos
    if ($first.Tag -eq 0xA0) { $pos += $first.Hdr + $first.Len }      # optional [0] version
    foreach ($skip in 1..5) {                                         # serial, sigalg, issuer, validity, subject
        $e = Read-Der $der $pos
        $pos += $e.Hdr + $e.Len
    }
    $spki = Read-Der $der $pos
    $bytes = $der[$spki.Pos..($spki.Pos + $spki.Hdr + $spki.Len - 1)]
    $sha = [Security.Cryptography.SHA256]::Create().ComputeHash([byte[]]$bytes)
    [Convert]::ToBase64String($sha)
}

# --- find Chrome ----------------------------------------------------------------
$candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$chrome = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) { throw "Chrome was not found. Install it or edit this script." }

$args = @(
    "--user-data-dir=$Profile",
    "--proxy-server=http://${ProxyHost}:$Port",
    "--no-first-run",
    "--no-default-browser-check"
)
if (-not $InstalledCa) {
    if (-not (Test-Path $CaFile)) { throw "CA file not found: $CaFile (use -CaFile, or -InstalledCa)" }
    $hash = Get-SpkiHash $CaFile
    $args += "--ignore-certificate-errors-spki-list=$hash"
    Write-Host "Trusting the FPGate CA only inside this Chrome (spki sha256/$hash)"
}
Write-Host "Chrome -> proxy ${ProxyHost}:$Port   profile: $Profile"
Write-Host "Everything else on this computer keeps going direct."
Start-Process -FilePath $chrome -ArgumentList ($args + $Url)
