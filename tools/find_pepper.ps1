# Find Pepper on the laptop's Wi-Fi network: the host with NAOqi's port 9559 open.
# Prints her IP (last line), or nothing. Called by tools/calib_session.sh.
$ip = Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi" -ErrorAction SilentlyContinue |
      Select-Object -First 1 -ExpandProperty IPAddress
if (-not $ip) { exit 1 }
$base = ($ip -split '\.')[0..2] -join '.'
Write-Host "searching $base.1-254 for port 9559..."
$t = @{}
foreach ($i in 1..254) {
    $h = "$base.$i"; $c = New-Object System.Net.Sockets.TcpClient
    $t[$h] = @($c, $c.ConnectAsync($h, 9559))
}
Start-Sleep -Milliseconds 3000
$found = $null
foreach ($h in $t.Keys) {
    $c, $task = $t[$h]
    if (-not $found -and $task.Status -eq 'RanToCompletion' -and $c.Connected) { $found = $h }
    $c.Close()
}
if ($found) { $found }
