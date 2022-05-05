# Install on Linux / OSx

Clone the repository and move into its root directory

```bash
git clone git@github.com:ucl-bug/jwave.git
cd jwave
```


## Install in an existing environment

Follow the instructions to install [Jax with CUDA support](https://github.com/google/jax#installation) if you want to use your GPU.

Then, simply install `jwave` using pip

```bash
pip install git+git@github.com:ucl-bug/jwave.git
```

## Install in a new environment

You can use the provided make file to generate an existing environment that contains `jwave`.

First, generate the environment using

```bash
make virtualenv
```

Then install jax with GPU support (assumes CUDA > 11.1) using

```bash
make jaxgpu
```

Before using `jwave`,  activate the environment from the root folder of jwave using 

```
sorce .venv/bin/activate
```

