# TASK 1 

## Introduction 
- Deadline: 14th of November, 15:00
- Meetings: 
  - Friday 28th of October, 15:00
  - Friday 4th of November, 15:00
  - Friday 11th of November, 15:00

## Task description
- Goal: PREDICT THE AGE OF A BRAIN FROM MRI FEATURES
- Data: MRI scans of the brain, composed of 832 features
  - Train: 1212 samples
  - Test: 776 samples

## Workflow 
- Preprocessing and Feature selection: Tristan
- Imputation and Outlier detection: Maja
- Regression: Yves 

## Installation
- Create a directory for the project with `mkdir Task1`
- Enter the directory with `cd Task1`
### GitHub 
- Initialize the repository with `git init`
- Add the remote repository with `git remote add origin https://github.com/New-T-T/AML_TASK1.git`
- Pull the repository with `git pull origin master`
### Python
- Create a virtual environment with `python3 -m venv venv`
- Activate the virtual environment with `source venv/bin/activate`
- Install the requirements with `pip install -r requirements.txt`


## Program 

### Preprocessing and Feature selection
- Preprocessing: 
  - Train, test split for both X and y
  - The following steps are applied only on X (according to this [link](https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re))
  - Remove features with more than 50% of missing values: TODO 
  - Imputing: using `SimpleImputer` from `sklearn.impute`
    - TODO: can be replaced by an iterative imputer 
  - Standardization: using `StandardScaler` from `sklearn.preprocessing`
  - Removing low variance features: using `VarianceThreshold` from `sklearn.feature_selection`
  - Removing correlated features : [Section: Removing Correlated Features](https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/)
- Feature selection: 
  - Sequential Feature Selection (SFS) using Random Forest Regressor : [medium](https://towardsdatascience.com/5-feature-selection-method-from-scikit-learn-you-should-know-ed4d116e4172)
    - WARNING: SUPER LONG TO RUN
- Outlier detection: 
  - UMAP: 
    - Stochastic process: fixing the seed 
    - official [documentation](https://umap-learn.readthedocs.io/en/latest/index.html)
    - Basic parameters for optimization: [Basic UMAP Parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)


## TODO
- Maja 
  - [ ] Add a simple explanation in the README and in the docstrings about outlier detection ("outlier score"???)
  - Outlier detection
    - [ ] Make it remove more than 2 outliers
    - [ ] Apply the same method on the TEST set
- Tristan
  - Feature selection: 
    - [ ] Think of better methods
- Yves
  - [ ] Grid search for the best parameters of the regressor