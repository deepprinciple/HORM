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
pip install -r requirements.txt
```



# How to run this code:

<<<<<<< HEAD
### Train models

To train a model, please select the desired architecture from the available options: 'LeftNet', 'EquiformerV2', and 'AlphaNet'.  Specify your choice in the `model_type` field within the `ft.py` file.

=======
### Train model 
>>>>>>> fab198f6bd695897606fbe276cfbe11155a22763

```shell
python ft.py
```


# License
This project is licensed under the Apache License 2.0. For more details about the Apache License 2.0, please refer to the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).
