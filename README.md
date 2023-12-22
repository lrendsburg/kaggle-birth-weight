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
├── dataset/                      # Datasets
│   ├── preprocessing_method_1/   # One folder per preprocessing method with data for each stage
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   └── X_test.npy
│   └── ...                       
├── notebooks/                    # Notebooks for exploration and analysis
├── predictions/                  # Prediction results for submission
└── src/                          # Source code
    ├── models/                   # One folder per model
    │   ├── model_1/
    │   └── ...
    ├── preprocessing/            # One .py file per preprocessing method
    |   ├── preprocessing_1.py
    |   └── ...
    └── utils/                    # Utility functions
```


## Workflow
TODO: write 

## Models

### Approach 1: Neural networks that predict the confidence interval
**Motivation**
- Dual output (interval mean and width) captures uncertainty to exactly the degree it is needed for this challenge
- Instance-specific estimation of uncertainty
- Can directly optimize the loss (Winkler interval score) that is used for evaluation

**Challenges**
- Unclear whether neural networks are competitive on structured data

### Approach 2: Traditional ML methods (random forests, gradient boosting, ...) that predict quantiles
**Motivation**
- Traditional methods generally perform better on structured data than deep learning methods
- Quantile regression captures uncertainty to exactly the degree it is needed for this challenge
- Instance-specific estimation of uncertainty
- For minimizing confidence interval size, can predict multiple quantiles (.01, ..., .09, .91, ..., .99) and then choose the 90% interval with the smallest size

**Challenges**
- No direct relation to the specific loss (Winkler interval score); might be suboptimal
- Method for miminizng confidence interval size is somewhat ad-hoc


### Approach 3: Traditional ML methods (random forests, gradient boosting, ...) with conformal prediction
**Motivation**
- Traditional methods generally perform better on structured data than deep learning methods
- Easy to use: out-of-the-box addon for black box prediction methods

**Challenges**
- Estimation of uncertainty is not instance-specific $\implies$ suboptimal if uncertainty depends on instances

### Approach 4: Bayesian methods
**Motivation**
- Most natural description of uncertainty

**Challenges**
- Unclear how to formulate a Bayesian model, little prior knowledge. Requires intensive feature-engineering and modeling.

### Approach 5: Bayesian Neural Networks (specifically the hybrid [Laplace Redux](https://proceedings.neurips.cc/paper_files/paper/2021/file/a7c9585703d275249f30a088cebba0ad-Paper.pdf))
**Motivation**
- All advantages of deterministic neural networks
- Hopefully captures uncertainty even better due to the added uncertainty through the Laplace approximation

**Challenges**
- Unclear whether neural networks are competitive on structured data
