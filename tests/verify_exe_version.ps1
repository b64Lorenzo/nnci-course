param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedVersion
)

$versionInfo = (Get-Item -LiteralPath $Executable).VersionInfo
if ($versionInfo.FileVersion -ne $ExpectedVersion) {
    throw "Expected FileVersion '$ExpectedVersion'; got '$($versionInfo.FileVersion)'"
}
if ($versionInfo.ProductVersion -ne $ExpectedVersion) {
    throw "Expected ProductVersion '$ExpectedVersion'; got '$($versionInfo.ProductVersion)'"
}
