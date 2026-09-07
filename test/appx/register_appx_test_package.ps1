# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$ExePath
)
$ErrorActionPreference = 'Stop'

$ExeFull = (Resolve-Path $ExePath).Path
$ExeDir = Split-Path $ExeFull
$ExeName = Split-Path $ExeFull -Leaf
$ExeBase = [IO.Path]::GetFileNameWithoutExtension($ExeName)

$PackageName = "mlvc-$($ExeBase -replace '_', '-')"
$DisplayName = $ExeBase
$AliasFullTrust = "app_${ExeBase}.exe"
$AliasAppContainer = "app_${ExeBase}_ac.exe"

# AppContainer cannot load package dependencies through symbolic links whose
# targets are outside the package ACL. Materialize linked DLLs in the loose
# package directory before registration.
Get-ChildItem -Path $ExeDir -Filter '*.dll' -File | Where-Object LinkType | ForEach-Object {
  $LinkPath = $_.FullName
  $TargetPath = $_.Target | Select-Object -First 1
  if (-not [IO.Path]::IsPathRooted($TargetPath)) {
    $TargetPath = Join-Path $_.DirectoryName $TargetPath
  }
  $TargetPath = (Resolve-Path $TargetPath).Path
  Write-Host "Materializing linked dependency $($_.Name) from $TargetPath"
  Remove-Item -LiteralPath $LinkPath -Force
  Copy-Item -LiteralPath $TargetPath -Destination $LinkPath
}

$PlaceholderLogoBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
$AssetsDir = Join-Path $ExeDir 'Assets'
New-Item -ItemType Directory -Force $AssetsDir | Out-Null
[IO.File]::WriteAllBytes((Join-Path $AssetsDir 'logo.png'), [Convert]::FromBase64String($PlaceholderLogoBase64))

$Manifest = (Get-Content "$PSScriptRoot\AppxManifest.template.xml" -Raw) `
  -replace '\{\{PACKAGE_NAME\}\}', $PackageName `
  -replace '\{\{DISPLAY_NAME\}\}', $DisplayName `
  -replace '\{\{EXE_NAME\}\}', $ExeName `
  -replace '\{\{ALIAS_FULL_TRUST\}\}', $AliasFullTrust `
  -replace '\{\{ALIAS_APP_CONTAINER\}\}', $AliasAppContainer

$ManifestPath = Join-Path $ExeDir 'AppxManifest.xml'
Set-Content -Path $ManifestPath -Value $Manifest -Encoding UTF8
$ExistingPackage = Get-AppxPackage -Name $PackageName
if ($ExistingPackage) {
  Write-Host "Removing existing registration for $PackageName"
  $ExistingPackage | Remove-AppxPackage
}
Add-AppxPackage -Register $ManifestPath -ForceApplicationShutdown

# Grant AppContainer read access to model bundles and test data. S-1-15-2-1/2 = ALL [RESTRICTED] APPLICATION PACKAGES.
$DataDirectories = @(
  $env:LIBMLVC_MODEL_BUNDLES_DIR,
  $env:LIBMLVC_TEST_DATA_DIR
)
foreach ($DataDirectory in $DataDirectories) {
  $DataDirectoryFull = (Resolve-Path $DataDirectory).Path
  Write-Host "Granting AppContainer read access to $DataDirectoryFull"
  & icacls $DataDirectoryFull /grant '*S-1-15-2-1:(OI)(CI)(RX)' /Q | Out-Null
  & icacls $DataDirectoryFull /grant '*S-1-15-2-2:(OI)(CI)(RX)' /Q | Out-Null
}

Write-Host "Registered $PackageName. Both aliases available from terminal:"
Write-Host "  $AliasFullTrust    (packaged, mediumIL)"
Write-Host "  $AliasAppContainer    (packaged, appContainer)"
Write-Host "To unregister: Get-AppxPackage $PackageName | Remove-AppxPackage"
