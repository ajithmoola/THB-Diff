# THB-DIFF: a GPU-accelerated differentiable programming framework for THB-spline.

Finished the important parts but lacks comments and documentation. This repository is still a work in progress.

The paper can be accessed [here](https://rdcu.be/dyLQl).

![image](https://github.com/ajithmoola/THB/assets/113499868/3bb2f1d4-de0a-4d86-a8f5-db4c7eab0466)

## Installation

- Install conda (Follow the instructions on this [webpage](https://docs.anaconda.com/free/miniconda/miniconda-install/)). Although, for MacOS it would be easier to install conda using Homebrew.

- Create a conda environment

    ```
    conda create -n THB python=3.10
    ```

- Activate the conda environment.

  ```
  conda activate THB
  ```

- Install the following dependencies using either ```pip``` or ```conda```
    - numpy
    - matplotlib
    - jax
    - jaxlib
    - tqdm
    - numba
    - pyvista


    ```
    pip install jax jaxlib numba matplotlib tqdm numba pyvista
    ```

    Replace jax and jaxlib with pytorch if a pytorch version is preferred.

- To access CUDA-accelerated kernels or c++ functions for THB-spline evaluation using PyTorch, run the following command from the THB_extensions directory.

    ```
    python setup.py build
    python setup.py install
    ```

- Install source code THB-Diff

    ```
    pip install .
    ```
    for editable installtion
    ```
    pip install -e .
    ```

## Citation
```
@Article{thbdiff,
author={Moola, Ajith and Balu, Aditya and Krishnamurthy, Adarsh and Pawar, Aishwarya},
title={THB-Diff: a GPU-accelerated differentiable programming framework for THB-splines},
journal={Engineering with Computers},
year={2023},}
```
