# Install on Windows

Installing `jwave` on Windows **without hardware acceleration** can be easily done using `pipy`, as in every other OS that has python installed. 

Installation **with CUDA support** or **CPU compilation** is mainly limited by the lack of `jaxlib` wheels for Windows (see [this issue](https://github.com/google/jax/issues/5795) for more information).

Therefore, `jwave` with GPU is only partially supported on Windows. We provide an experimental way of installing `jwave` on Windows with GPU support using the Windows Subsystem for Linux.  

Alternatively, one can [build JAX from the sources](https://jax.readthedocs.io/en/latest/developer.html) or, if you already have a python environment setup, install [those unofficial wheels](https://github.com/cloudhan/jax-windows-builder).

Any help to improve the installation on Windows is more than welcome 😊.

**Tested on**:

- OS: Windows 10
- GPU: Nvidia GTX 1060

## Install

Open a `PowerShell` instance and paste this

```ps
$JwaveWinInstaller = Invoke-WebRequest https://raw.githubusercontent.com/ucl-bug/jwave/main/scripts/jwave_win_install.ps1
Invoke-Expression $($JwaveWinInstaller.Content)
```

then follow the instructions to install `jwave`.

## Uninstall

If you have installed `jwave` using the script above, open a `PowerShell` instance and type

```ps
scoop uninstall scoop
```

to completely remove jwave and all other programs installed (e.g. the virtual machine)
