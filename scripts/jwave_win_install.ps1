# Initial info.
Write-Output "jwave Windows installer"
Write-Output "========================"
Write-Output "This script will install jwave on your Windows machine with CPU/GPU acceleration."
Write-Output "It is recommended to use this script if you are a beginner with Python."
Write-Output ""
Write-Output "If you don't need hardware acceleration, we recommend to install jwave with pip on an existing python environment, rather than using this script."
Write-Output "Alternatively, you can build jax (with/without GPU support) from source before installing jwave, see: https://jax.readthedocs.io/en/latest/developer.html,"
Write-Output "or use the unofficial windows wheels provided in https://github.com/cloudhan/jax-windows-builder"
Write-Output ""
Write-Output "This script will install jwave on your Windows machine using the Windows Subsystem for Linux (WSL)."
Write-Output ""

$continue = Read-Host "Do you want to continue? (Y/N)"
if ($continue -ne "Y") {
    Write-Output "Aborting installation."
    exit
}

# Installing scoop if doesn't exists
if (Get-Command scoop -ErrorAction SilentlyContinue) {
    Write-Output "scoop is already installed."
} else {
    Write-Output "-- jwave installer: Installing scoop. Press 'Y' when prompted to install scoop." 
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    Invoke-WebRequest get.scoop.sh | Invoke-Expression
}

# Installing Git
Write-Output "-- jwave installer: Installing Git. Press 'Y' when prompted to install Git."
scoop bucket add extras
scoop install git

# Cloning jwave
$confirm = Read-Host "-- jwave installer: Do you want to clone jwave in this folder? (Y/N)"
if ($confirm -eq "Y") {
    Write-Output "-- jwave installer: Cloning jwave in this folder."
    git clone  git@github.com:ucl-bug/jwave.git
    Set-Location jwave
}
else {
    Write-Output "-- jwave installer: Aborting. Please run this script again from the folder where you want to clone jwave."
    exit
}
    
# Installing Manjaro WSL
Write-Output "-- jwave installer: Installing Git and Windows Subsystem for Linux (Manjaro)"
scoop install manjarowsl

# Inform user of next steps
Write-Output "-- jwave installer: Windows Subsystem for Linux (WSL) is installed."
Write-Output "                    In the next steps, you'll have to finalize its setup."
Write-Output "                    Follow the instructions on the screen to resize the WSL"
Write-Output "                    and create a new user."

wsl

Write-Output "-- jwave installer: WSL is configured. If an extra WSL window has appeared, please close it."

Write-Output "-- jwave installer: Updating WSL."
wsl sudo pacman -Syyu --noconfirm dos2unix
wsl dos2unix ./scripts/jwave_install_wsl_gpu.sh
wsl dos2unix ./scripts/jwave_install_wsl_cpu.sh

# Checking installation type
$install_type = Read-Host "-- jwave installer: Do you want to install jwave with CPU-only support or with GPU support? (CPU/GPU)"
if ($install_type -eq "CPU") {
    Write-Output "-- jwave installer: Installing jwave with CPU support."
    wsl chmod +x ./scripts/jwave_install_wsl_cpu.sh
    wsl ./scripts/jwave_install_wsl_cpu.sh
}
elseif ($install_type -eq "GPU") {
    Write-Output "-- jwave installer: Installing jwave with CPU support."
    Write-Output "                    The script assumes that you have an NVIDIA GPU card"
    wsl chmod +x ./scripts/jwave_install_wsl_gpu.sh
    wsl ./scripts/jwave_install_wsl_gpu.sh
}
else {
    Write-Output "-- jwave installer: Unknown installation type. Aborting."
    exit
}

Write-Output "========================"
Write-Output "jwave is installed."