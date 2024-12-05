# Factorizer library
This repository contains a series of factorizers: 
1. In-memory factorizer presented in Nature Nanotechnology: [In-memory factorization of holographic perceptual representations](https://www.nature.com/articles/s41565-023-01357-8)
2. Block codes factorizer presented in Neurosymbolic Artificial Intelligence: [Factorizers for distributed sparse block codes](https://content.iospress.com/download/neurosymbolic-artificial-intelligence/nai240713?id=neurosymbolic-artificial-intelligence%2Fnai240713)
3. Asymmetric factorizer presented at MLNCP workshop at NeurIPS 2024: [On the Role of Noise in Factorizers for Disentangling Distributed Representations](https://openreview.net/forum?id=VYryqVqQEF)


Authors: Michael Hersche <michael.hersche@ibm.com>, Geethan Karunaratne <kar@zurich.ibm.com>, Aleksandar Terzic <aleksandar.terzic1@ibm.com>

#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `environment.yml`. 
You can recreate our environment by running 

```
$ conda create --name IMFEnv python=3.11
$ conda activate IMFEnv
```
Install pytorch CUDA and other requirements
```
$ (IMFEnv) conda install pytorch torchvision cudatoolkit=12.1 -c pytorch -c nvidia
$ (IMFEnv) pip install -r requirements.txt
```

#### Run tests

To run the factorizer execute the following command from the root of the repository.
```
$ (IMFEnv) python main_capacity.py --custom-config experiments/<Experiment>/config.json
```

Experiment configurations:

| Experiment        | Description                         |  Ref  |
|-------------------|-------------------------------------|-------|
| 100a_baseline     | Baseline factorizer                 |       |
| 100b_dense        | Standard factorizer                 |  [1]  |
| 100e_totnoise     | Factorizer total noise sweep        |  [1]  |
| 100e_prnoise      | Factorizer programming noise sweep  |  [1]  |
| 100e_rdnoise      | Factorizer read noise sweep         |  [1]  |
| 200a_bcf          | Block codes factorizer              |  [2]  |
| 300a_assymetric_cb | Asymmetric codebook factorizer      |  [3]  |

The script saves a `.npz` file which you can load after running and plot the accuracy vs. problem size. 

#### Citation
Please cite the corresponding work when using one of our factorizer models: 

[1]
```
@Article{langenegger2023imfactorizer,
    Author = {Langenegger, Jovin and Karunaratne, Geethan and Hersche, Michael and  Benini, Luca and Sebastian, Abu and Rahimi, Abbas },
    Journal = {Nature Nanotechnology},
    Year = {2023},
    Title = {In-memory factorization of holographic perceptual representations}
    }
```
[2]
```
@Article{hersche2024factorizers,
    Author = {Hersche, Michael and Terzi{\'c}, Aleksandar and Karunaratne, Geethan and Langenegger, Jovin and Pouget, Ang{\'e}line and Cherubini, Giovanni and Benini, Luca and Sebastian, Abu and Rahimi, Abbas},
    Journal = {Neurosymbolic Artificial Intelligence},
    Year = {2024},
    Title = {Factorizers for distributed sparse block codes}
    }
```
[3]
```
@inproceedings{
karunaratne2024acfactorizer,
title={On the role of noise in factorizers for disentangling distributed representations},
author={Geethan Karunaratne and Michael Hersche and Abu Sebastian and Abbas Rahimi},
booktitle={NeurIPS 2024 Workshop Machine Learning with new Compute Paradigms},
year={2024},
}
```

#### License
Our code is licensed under Apache 2.0. Please refer to the LICENSE file for the licensing of our code.

