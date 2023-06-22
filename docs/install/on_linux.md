# Install on Linux / OSx

## Install in an existing environment

Follow the instructions to install [Jax with CUDA support](https://github.com/google/jax#installation) if you want to use your GPU.

Then, simply install `jwave` using pip

```bash
pip install jwave
```

## Install automatically in a new environment

You can use the provided make file to generate an existing environment that contains `jwave`.

First, clone the repository and move into its root directory

```bash
git clone git@github.com:ucl-bug/jwave.git
cd jwave
```

Then, generate the environment using

```bash
make virtualenv
```

Install jax with GPU support (assumes CUDA > 11.1) using

```bash
make jaxgpu
```

Before using `jwave`,  activate the environment from the root folder of jwave using

```
source .venv/bin/activate
```
