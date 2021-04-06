# Install
If not already done, [install Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (*not Miniconda*).

Generate a new environment with
```bash
conda create --name myenv python=3.8
```
then activate it
```bash
conda activate myenv
```

Before installing `jwave`, make sure that [you have installed JAX](https://github.com/google/jax#installation). Follow the instruction to install JAX with NVidia GPU support if you want to use `jwave` on the GPUs: the version of cuda is 10.2. 

Install jwave by `cd` in the repo folder an run
```bash
pip install -r requirements.txt
pip install -e .
```

If you want to run the notebooks, you should also install the following packages
```bash
pip install jupyter, tqdm
```

## GPU support on `kinsler`
On `kinsler` a few extra steps are needed to have JAX working on the GPU because of the non-standard CUDA installation. First load CUDA, then tell XLA where to find the CUDA drivers and restrict the number of visible devices
```bash
module load CUDA
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/software/CUDA/10.2.89"
export CUDA_VISIBLE_DEVICES="0,1"
```
The last step is optional, but by default `jax` occupies all available devices at startup. Make visibles only the GPUs you are going to need to preventing blocking access to the remaining devices for other users.

If you don't want to run those line everytime a new bash session is opened, you can paste them in your `~/.bashrc` file. 

Alternatively, add the following lines at the top of your main python script / notebook
```python
import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/apps/software/CUDA/10.2.89"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
```
However, you still need to run `module load CUDA` from the terminal before running any python script to use the GPU (I suggest adding it to the `.bashrc` file). 

## Use kWave in python
Lots of functionality is still missing compared to `kWave`, expecially in terms of functions constructing objects. If you want to use `kWave` methods from python, you should install the MATLAB engine for python. On `kinsler`, this is done as follows:
```bash
conda activate myenv
module load Matlab/2019a
cd /apps/software/Matlab/R2019a/extern/engines/python/
python setup.py build --build-base=$(mktemp -d) install
```