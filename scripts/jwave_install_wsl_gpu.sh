# Installing prerequisites for JWave on WSL
echo "-- jwave installer: Installing packages, please type your password when prompted"
sudo pacman -Syyu --noconfirm pamac base-devel git cudnn python3

# Source bashrc
source ~/.bashrc

# Making virtual environment
echo "-- jwave installer: Making virtual environment and installing jax[gpu]"
make virtualenv
make jaxgpu


# Add a line to the .bashrc file that goes to jwave and activates the environment at startup
CURRENT_DIR=$(pwd)
echo "source $CURRENT_DIR/.venv/bin/activate" >> ~/.bashrc
echo "cd $CURRENT_DIR" >> ~/.bashrc

# Inform the user
echo "jwave is installed! To start using jwave, open a new Power Shell and type 'wsl'"

# Exit wsl
exit