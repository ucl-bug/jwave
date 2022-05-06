Write-Output "-- Uninstalling jwave"

# Remove scoop
scoop uninstall scoop

# Remove jwave folder if exists
Remove-Item -Path "~/.jwave" -Force -Recurse -ErrorAction SilentlyContinue

Write-Output "-- jwave uninstalled"