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

## Example Usage

Create B-spline objects which constitute the initial tensor-product

For 2D THB-splines

```
from THB.datastructures import BSpline, TensorProduct

bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1]), degree=2)
bs2 = BSpline(knotvector=np.array([0, 0, 0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 1, 1, 1], degree=3)

tp_2D = TensorProduct([bs1, bs2])
```

For 3D THB-splines

```
from THB.datastructures import BSpline, TensorProduct

bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1]), degree=2)
bs2 = BSpline(knotvector=np.array([0, 0, 0, 0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 1, 1, 1], degree=3)
bs3 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)

tp_3D = TensorProduct([bs1, bs2, bs3])
```
THB-Diff supports non-uniformly spaced knots as input knotvectors for constituent ```BSplines```. The ```TensorProduct``` can have ```BSplines``` objects of varied degrees and knotvectors as input for tensor-product construction.

Create a ```Space``` object which handles the low-level queries and operations on THB-spline domain and function space, and a ```THB``` object to handle high-level operations, GPU-accelerated THB-spline evaluation methods, loss functions, gradient computation and other utilites.

```
h_space = Space(tensor_product=tp, num_levels=3)
THB = THB_layer(h_space)
```

_documentation work in progress..._

## Citation
```
@Article{thbdiff,
author={Moola, Ajith and Balu, Aditya and Krishnamurthy, Adarsh and Pawar, Aishwarya},
title={THB-Diff: a GPU-accelerated differentiable programming framework for THB-splines},
journal={Engineering with Computers},
year={2023},}
```
