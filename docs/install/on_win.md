# Install on Windows

`jwave` is based on JAX, which currently offers [limited support for Windows platforms](https://github.com/google/jax#installation) (see [this issue](https://github.com/google/jax/issues/5795) for more information).

Following are a few workarounds for running `jax` (and, therefore, `jwave`) on a Windows machine:

1. [Install on an isolated WSL](#install-on-an-isolated-wsl)
2. [Install using the unofficial jax wheels](#install-using-the-unofficial-jax-wheels)
3. [Building jax from source](#building-jax-from-source)

Before running any of them, please make sure that your Windows machine is up-to-date, by clicking on `Start`, then typing `Check for updates` in the search bar and pressing enter. Install any updates that are available.

Any help to improve the installation on Windows is more than welcome 😊.

<br/>

## Install on an isolated WSL

This is the easiest option. It uses the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) (WSL) to install an [Arch-based Linux distribution](https://github.com/sileshn/ManjaroWSL), in which python and the required packages are installed.

The WSL and other required programs are installed using [`scoop`](https://scoop.sh/) without Adiministrator privileges, which ensures that they are isolated from the rest of the system and minimizes the risk of interference with other programs.

The WSL shares the same filesystem as the host machine, therefore all files can be accessed directly from the Windows File Explorer. For developing your Python code, we recommend to use Visual Studio Code with the [Remote - WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) extension: see [this page](https://code.visualstudio.com/docs/remote/wsl-tutorial) for a detailed tutorial.

### Prerequisites

You must be running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11.

Make sure that the Windows Subsystem for Linux feature is enabled in your system. Click on `Start`, type `PowerShell`, then right-click on `Windows PowerShell` and click `Run As Administrator`.

In the PowerShell window, type the following command and press `Enter`:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

Similarly, check that the Virtual Machine feature is enabled by typing the following command and pressing `Enter`:

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Close the PowerShell window and restart your computer.

The next step is to make sure that your system has the latest Linux kernel update, by downloading and installing [this package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi).

### How to install

Open a `PowerShell` instance and paste this

```ps
$JwaveWinInstaller = Invoke-WebRequest https://raw.githubusercontent.com/ucl-bug/jwave/main/scripts/jwave_win_install.ps1
Invoke-Expression $($JwaveWinInstaller.Content)
```

then follow the instructions to install `jwave`.

### How to uninstall

If you have installed `jwave` using the script above, open a `PowerShell` instance and type

```ps
scoop uninstall scoop
```

to completely remove jwave and all other programs installed (e.g. the virtual machine).

<br/>

## Install using the unofficial jax wheels

⚠️ This method uses a [community supported Windows build for jax](https://github.com/cloudhan/jax-windows-builder), which is in alpha state and is not guaranteed to work. Only CPU and CUDA 11.x are supported.

### Prerequisites
This method assumes that you've aready setup a Python environment in your Windows machine. We recommend to use Anaconda to keep your Python installation separate from the rest of your system: [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) offers a relatively quick guide to Anaconda installation and how to manage environments.

### How to install
After activating your python environment, follow [the README](https://github.com/cloudhan/jax-windows-builder/blob/main/README.md) to install `jax` for your python and (if needed) CUDA version.

Then install `jwave` using

```powershell
pip install git+https://github.com/ucl-bug/jwave`
```

<br/>

## Building jax from source
The latest method is to build `jax` from source. This is not recommended for users that are not familiar with building large software packages from source.

### How to install
Follow the [guide on the jax docs](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-jaxlib-from-source-on-windows) for building `jax` from source, up to the end of the *"Running the tests"* section

Then install `jwave` using

```powershell
pip install git+https://github.com/ucl-bug/jwave
```
