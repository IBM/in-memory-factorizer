# In-memory factorizer
This is the repository for the code associated with the paper titled "In-memory factorization of holographic perceptual representations"

Authors: Michael Hersche <her@zurich.ibm.com>, Geethan Karunaratne <kar@zurich.ibm.com>

#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `environment.yml`. 
You can recreate our environment by running 

```
$ conda create --name IMFEnv python=3.6
$ conda activate IMFEnv
```
Install pytorch CUDA and other requirements
```
$ (IMFEnv) conda install pytorch=1.3 torchvision cudatoolkit=10.1 -c pytorch
$ (IMFEnv) pip install -r requirements.txt
```

#### Run tests

To run the factorizer execute the following command from the root of the repository.
```
$ (IMFEnv) python main_capacity.py --custom-config experiments/<Experiment>/config.json
```

Available experiments are:

| Experiment    | Description                         |
|---------------|-------------------------------------|
| 100a_baseline | Baseline factorizer                 |
| 100b_dense    | Standard factorizer                 |
| 100e_totnoise | Factorizer total noise sweep        |
| 100e_prnoise  | Factorizer programming noise sweep  |
| 100e_rdnoise  | Factorizer read noise sweep         |


The script saves a `.npz` file which you can load after running and plot the accuracy vs. problem size. 

#### Citation
```
@Article{langenegger2023imfactorizer,
    Author = {Langenegger, Jovin and Karunaratne, Geethan and Hersche, Michael and  Benini, Luca and Sebastian, Abu and Rahimi, Abbas },
    Journal = {Nature Nanotechnology},
    Year = {2023},
    Title = {Constrained Few-shot Class-incremental Learning},
    Year = {2022}}
```

#### License
Our code is licensed under Apache 2.0. Please refer to the LICENSE file for the licensing of our code.

