# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

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
