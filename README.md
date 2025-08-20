# Learning from Complementary Features


## requirements

* Python: 3.9.14

* Library: Please see `requirements.txt`




## Directory Structure

```
CFL_benchmarks_public/
  ├─README.md
  ├─requirements.txt
  ├─exp1
  │  ├─main.py
  │  ├─pred_avoid_est.py
  │  ├─config
  │  │  ├─...
  │  └─shells
  │     ├─...
  ├─exp2
  │  ├─iterative_learning.py
  │  ├─joint_learning.py
  │  ├─config
  │  │  ├─...
  │  ├─shells_iterative
  │  │  ├─...
  │  └─shells_joint
  │     ├─...
  ├─libs
  │  ├─cfl_learning.py
  │  ├─evaluation.py
  │  ├─helpers.py
  │  ├─learning.py
  │  ├─load_data.py
  │  ├─nn_models.py
  │  ├─utils.py
  │  ├─utils_processing.py
  │  └─existing_methods
  │     └─nn_based
  │        ├─idgp.py
  │        ├─nn_models.py
  │        ├─nn_utils.py
  │        ├─nn_loss.py
  │        └─risk_based.py
  ├─data
  │  ├─adult
  │  │  └─nn_based
  │  ├─bank
  │     └─nn_based
```

* `exp1`: This directory includes the experimental codes for sequential learning strategy
    * `main.py`: This is the experiment script.
    * `shells/`: Shell scripts for experiments
    * `config/`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.

* `exp2`: This directory includes the experimental codes for iterative and joint learning strategies
    * `iterative_learning.py`: This is the experiment script for iterative learning strategy.
    * `joint_learning.py`: This is the experiment script for joint learning strategy.
    * `shells/`: Shell scripts for experiments
    * `config/`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.

* `libs`: This is a directory that stores functions and other utilities used in `exp1/` and `exp2/`.
* `config`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.
* `data`: This is a directory that stores the datasets used in the experiments. Each data must be downloaded from [UCI repository](https://archive.ics.uci.edu/).


## Datasets download Links

* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  * Unzip `bank+marketing.zip`, and place all the files inside `bank.zip` into `CFL_benchmarks_public/data/bank/`.

* [Adult](https://archive.ics.uci.edu/dataset/2/adult)
  * Unzip `adult.zip`, and place all the files located directly under it into `CFL_benchmarks_public/data/adult/`.

## Example of Experiment Execution

```bash

cd exp1
python ./shells/cc/adult_all.sh


```
