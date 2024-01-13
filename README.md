# Submission for Kaggle competition [Prediction interval competition I: Birth weight](https://www.kaggle.com/competitions/prediction-interval-competition-i-birth-weight)

## Setup
1. **Clone the Repository:** First, clone the project repository to your local machine.
    ```bash
    git clone https://github.com/lrendsburg/kaggle-birth-weight.git
    cd kaggle-birth-weight
    ```

2. **Install the Project:** You'll need to install the project in editable mode to make imports work. Run the following command in the root directory of the project:
    ```bash
    pip install -e .
    ```

## Repository structure
``` bash
.
├── conf/                         # Configuration files for hyperparameter tuning
├── datasets/                     # Datasets
│   ├── preprocessing_method_1/   # Data processed with method 1
│   │   ├── X_train.npy           # Training features
│   │   ├── y_train.npy           # Training labels
│   │   ├── X_val.npy             # Validation features
│   │   ├── y_val.npy             # Validation labels
│   │   └── X_test.npy            # Test features
│   └── ...                       # Other preprocessing methods
├── notebooks/                    # Jupyter notebooks for exploration and analysis
├── predictions/                  # Prediction results for submission
├── reports/                      # Generated reports and experiment summaries
└── src/                          # Source code
    ├── models/                   # Model-specific code
    │   ├── model_1/              # Model 1 implementation
    │   │   └── main.py           # Main script for running model 1 experiments
    │   └── ...                   # Other models
    ├── prediction/               # Prediction-related functions
    │   └── prediction.py         # Prediction function implementations
    ├── preprocessing/            # Preprocessing scripts
    │   ├── preprocessing_1.py    # Preprocessing method 1
    │   └── ...                   # Other preprocessing methods
    ├── experiment_pipeline.py    # Main experiment pipeline
    └── utils/                    # Utility functions and helper scripts

```


## Workflow
1. **Train-val split:** make train-val split by running `src/preprocessing/train_val_split.py`
2. **Preprocessing:** specify a preprocessing method in `src/preprocessing/` as an sklearn pipeline and use `src/utils/preprocessing.process_data` to generate the processed data under `datasets/`. See `src/preprocessing/simple_preprocessing.py` for an example.
3. **Model training:** specify one main file per model in `src/models/`. The model should be implemented as a class that inherits from `src/experiment_pipeline.BaseExperiment` and should provide the methods `fit`, `predict`, and `get_params`. Running the `run_experiment` method then runs the main experiment pipeline. See `src/models/dummy/main_constant.py` for an example.

### Combined methods: base model + prediction head
The following separation of base predictors and prediction heads increases modularity, as many base predictors can be combined with the same heads.

For methods that don't predict the confidence interval directly, but instead train on a surrogate objective, the prediction logic can be outsourced to `src/prediction/prediction.py`. So instead of implementing `predict` directly, a model can provide a `forward` method and inherit from a child of `src/prediction/prediction.Prediction`, which then provides the `predict` method implicitly. For example, the model in `src/models/xgboost/main_xgboost_conformal.py` only provides its point prediction in the `forward` and then inherits from `src/utils/prediction.ConformalPrediction` to get the `predict` method. 

### Hyperparameter sweeps
`optuna` is used for hyperparameter sweeps. This is done by creating an optuna study in the `main.py` of the model and then optimizing over the study in the `run_experiment` method. 
- The optimization objective is the Winkler interval score over the validation set for all methods except Bayesian ones, where the evidence is used instead.
- Search spaces for hyperparameters of models and prediction heads are outsourced to `conf/` 

### Logging
- Model parameters and train/val metrics are logged to mlflow. This is done automatically by the `BaseExperiment.run_experiments`
- For neural networks, the training progress is logged to wandb. 

### Reports
The folder `reports/` contains a report for each experiment setup in the form of `.md` files (based on the metadata logged to mlflow).

## Models

### Base architectures
xgboost / catboost / lightgbm / random forest / fully connected neural networks

### Prediction heads
- **Quantile regression** (predict a set of quantiles, e.g. [.03, .05, .07, .93, .95, .97]) and choose the 90%CI with the smallest size as prediction
- **Conformal prediction** *(is it actually called conformal in this context?)*
  - Augment point prediction $f(x)$ with prediction interval $f(x) + CI_\text{val}$ where $CI_\text{val}$ is chosen based on validation error
  - Mostly w.r.t square loss due to efficiency reasons (except for NNs), but MAE might me more appropriate considering how the Winkler interval score punishes prediction error linearly
- Two special setups for neural networks:
    - **Direct**: directly predicting mean and width of the prediction interval under Winkler interval score
    - **Bayesian**: Train usual regression, but throw away head (=deterministic linear regression) and replace it with a Bayesian linear regression head to get a posterior predictive in closed form

- Conformal prediction yields CI that are independent of $x$, while the other methods yield $x$-dependent CI. Whether this is an advantage depends on the variability of $y|x$ in $x$.

### Combinations
- $\{\text{xgboost}, \text{catboost}, \text{lightgbm}, \text{random forest}\} \times \{\text{conformal}, \text{quantile}\}$
- $\{\text{neural networks}\} \times \{\text{conformal}, \text{direct}, \text{bayes}\}$

## Open ends
- Spend more time on feature pre-processing
- Increase search space for neural networks, including techniques like dropout
- Get `DirectNN` to learn