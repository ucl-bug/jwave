sudo pacman -Syyu pamac base-devel git cudnn python3
git clone git@github.com:ucl-bug/jwave.git
cd jwave
make virtualenv
make jaxgpu

# Add a line to the .bashrc file that goes to jwave and activates the environment at startup
echo "source ~/jwave/bin/activate" >> ~/.bashrc
echo "cd ~/jwave" >> ~/.bashrc

echo "jwave is installed! To start using jwave from Windows, open a new Power Shell and type 'wsl'"