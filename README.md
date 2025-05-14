# HORM：A Molecular Hessian Database for Optimizing Reactive Machine Learning Interatomic Potentials

This is the official implementation for the paper: "A Molecular Hessian Database for Optimizing Reactive Machine Learning Interatomic Potentials". 

Please read and cite these manuscripts if using this example: XXX


- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [How to run this code](#how-to-run-this-code)

# Overview
Transition state (TS) characterization is central to computational reaction modeling, yet conventional approaches depend on expensive density functional theory (DFT) calculations, limiting their scalability. Machine learning interatomic potentials (MLIPs) have emerged as a promising approach to accelerate TS searches by approximating quantum-level accuracy at a fraction of the cost. However, most MLIPs are primarily designed for energy and force prediction, thus their capacity to accurately estimate Hessians, which are crucial for TS optimization, remains constrained by limited training data and inadequate learning strategies. This work introduces the Hessian dataset for Optimizing Reactive MLIP (HORM), the largest quantum chemistry Hessian database dedicated to reactive systems, comprising 1.84 million Hessian matrices computed at the $\omega$B97X/6-31G(d) level of theory. To effectively leverage this dataset, we adopt a Hessian-informed training strategy that incorporates stochastic row sampling, which addresses the dramatically increased cost and complexity of incorporating second-order information into MLIPs. Various MLIP architectures and force prediction schemes trained on HORM demonstrate up to a 63\% reduction in Hessian mean absolute error and up to a 200× increase in TS search success rates compared to models trained without Hessian information. These results highlight how HORM addresses critical data and methodological gaps, enabling the development of more accurate and robust reactive MLIPs for large-scale reaction network exploration.


# Installation Guide:


```shell
pip install .
```

Note: For torch-cluster installation, you need to install the version that matches your CUDA version. 
For example, if you encounter CUDA-related errors, you can uninstall torch-cluster and install the version matching your CUDA version. For CUDA 12.1:

```shell
pip uninstall torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```



# How to run this code:

### Train models

To train a model, please select the desired architecture from the available options: 'LeftNet', 'EquiformerV2', and 'AlphaNet'.  Specify your choice in the `model_type` field within the `ft.py` file.


```shell
python train.py
```

### Evaluate models

To evaluate a model, please specify the lmdb dataset and checkpoint, and run the following command:

```shell
python eval.py
```

# How to get dataset and checkpoints

The HORM dataset is available at: https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data

Pre-trained model checkpoints can be downloaded from: https://huggingface.co/yhong55/HORM

...


# License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For more details, please refer to the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
