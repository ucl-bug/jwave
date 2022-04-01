# Install on Windows

Installing `jwave` on Windows **without CUDA support** can be easily done using `pipy`, as in every other OS that has python installed. 

Installation **with CUDA support** is mainly limited by the lack of `jaxlib` wheels for Windows (see [this issue](https://github.com/google/jax/issues/5795) for more information).

Therefore, `jwave` on the GPU is only partially supported. This guide provides a potential way for installing `jwave` on Windows with GPU support using the Windows Subsystem for Linux.  

Any help from the community to improve the installation on Windows is more than welcome 😊.

**Tested on**:

- OS: Windows 10
- GPU: Nvidia GTX 1060


## Install and setup the WSL

(Skip this part if you already have WSL installed)

1. Download the latest zip release of ManjaroWSL [at this page](https://github.com/sileshn/ManjaroWSL/releases)
2. Extract the contents of the zip file
3. Double click on `Manjaro.exe`. If you get a warning from Windows Defender, click on `More info` and then on `Run Anyway`. This will install the Manjaro WSL.
4. Once the installation is completed, click on `Start`, type `wsl` and click on `wsl`.
5. Once in Manjaro, follow the default configuration (and generate a new user).

## Install the dependencies

Click on `Start`, type `Manjaro.exe` and press enter. 

Once logged in, install `cuda`, `python` and all the other dependencies using

```bash
sudo pacman -Syyu pamac base-devel git cudnn python3
```

Because we areinstalling packages, the command above requires administrator privilegies on the WSL. Therefore, when prompted, insert your password and answer `Y` to the confirmation prompts.

After the packages are installed, restart the WSL by closing and reopening to make the changes effective.

### 🔁 Tip

To access the WSL file system from Windows, type `explorer.exe .` from the terminal. This will open a `File Explorer` window.

I suggest two ways of programming in python via the WSL while having minimal compatibility issues:

1. Use [Visual Studio Code](https://code.visualstudio.com), which has got native support for [WSL developement](https://code.visualstudio.com/docs/remote/wsl#_getting-started) via the [Remote Developement extension pack](https://code.visualstudio.com/docs/remote/wsl#_getting-started)
2. Use [jupyter-lab](https://jupyter.org/install)

## Install `jwave`

Clone the repository and move into its root directory

```bash
git clone git@github.com:ucl-bug/jwave.git
cd jwave
```

You can use the provided make file to generate an existing environment that contains `jwave`. First, generate the environment using

```bash
make virtualenv
```

Then install jax with GPU support using

```bash
make jaxgpu
```

Before using `jwave`,  activate the environment from the root folder of jwave using 

```
sorce .venv/bin/activate
```