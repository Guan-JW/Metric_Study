# A Systematic Study on Early Stopping Criteria in HPO and the Implication of Uncertainty [EAB]

This paper delves into the pivotal role that early stopping criteria play in hyperparameter optimization (HPO). We introduce criteria that incorporate uncertainty and substantiate their practical significance through empirical experimentation. By shedding light on the influence of criterion selection, this research offers valuable insights for enhancing the effectiveness of HPO.

All the code used for empirical experiments, significance tests, and visualization, as discussed in the paper, is available in this repository for validation and replication.


## Installation

### Benchmark Setup
1. Nas-Bench-201 and HPOBench
    
    We make use of the high-level APIs defined in [HPOBench](https://github.com/automl/HPOBench) to access Nas-Bench-201. Here, we provide straightforward instructions on how to install it:
    ```bash
    git clone https://github.com/automl/HPOBench.git
    cd HPOBench
    pip install .
    ```

2. LCBench
    ```bash
    git clone https://github.com/automl/LCBench.git
    cd LCBench
    cat requirements.txt | xargs -n 1 -L 1 pip install
    ```

## Obtaining Results

The code for running the Hyperband algorithm on each benchmark to assess different criteria is available in separate scripts within the directory, named `LCBench_{dataset}.py` and `NB201_{dataset}.py`. The criteria for comparison are enumerated in the `utils.py` file, and the <em>--task</em> argument instructs the script on which criteria to evaluate. If no arguments are specified, the script defaults to comparing the commonly used criteria, i.e. <em>training loss/accuracy</em> and <em>validation loss/accuracy</em>. 
Each test runs 1000 repetitions by default, and the results are stored in the `Records/` directory. 

Run the tests as follows:

```bash
python NB201_ImageNet.py \
        --task acc_loss \               # Indicate which set of criteria to compare
        --rounds 1000 \                 # The number of repetitions for each comparison
        --max_iters 50 81 \  # The budget constraint R for Hyperband
        --eta 3                         # The filtering ratio for Hyperband
```

## Significance Testing

To compare the outcomes of each criterion, we employ the Wilcoxon signed-rank test to determine the presence of significant differences among the criteria. We also keep track of the number of wins, ties, and losses of each criterion across the 1000 repetitions. The scripts for these tests are located in the `Tests/` directory.

You can reproduce the comparison results by running:

```bash
cd Tests/
python wilcoxon.py \                    # Wilcoxon signed-rank test
        --task acc_loss \               # Indicate which set of criteria to compare
        --dataset ImageNet \            # Benchmark name
        --max_iters 50 81 \  # The budget constraint R for Hyperband
        --eta 3                         # The filtering ratio for Hyperband

python wtl.py \                         # Count the wins/ties/losses
        --task acc_loss \               
        --dataset ImageNet \            
        --max_iters 50 81\  
        --eta 3                         
```


## Visualization

Scripts for reproducing the figures presented in the paper are provided in the `zFigures/` directory. Simply execute the scripts to obtain visualization results.

## 
**Copyright**:  The copyright of this repository belongs to the authors of the VLDB'2025 paper submission (#345). The purpose of this package is only for the assessment by the VLDB'2025 program committee during the paper review process; any other uses for any other purposes are prohibited.