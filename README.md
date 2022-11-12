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
  - Lasso regression: using `LassoCV` from `sklearn.linear_model`
    - General idea: 
      - Lasso regression is a linear model that estimates sparse coefficients
      - The LassoCV object performs Lasso regression with built-in cross-validation
      - The optimal value of the regularization parameter is selected by cross-validation
      - The coefficients of the model are then used to select the most important features
    - Sklearn hyperparameters: 
      - `alphas`: array of alpha values to try
        - Default: not clear
        - TODO: uderstand
      - `eps`: float, optional
        - Length of the path. `eps=1e-3` means that `alpha_min / alpha_max = 1e-3`.
        - Default: `1e-3`
      - `n_alphas`: int, optional
        - Number of alphas along the regularization path
        - Default: `100`
      - `cv`: number of folds in cross-validation
      - `max_iter`: maximum number of iterations to perform
      - `tol`: tolerance for the optimization
    - Nb of simulations: 
      - CV * n_alphas  
        - default = 5 * 100 = 500
    - links 
      - https://stats.stackexchange.com/questions/68562/what-does-it-mean-if-all-the-coefficient-estimates-in-a-lasso-regression-converg
      - 
  - Sequential Feature Selection (SFS) using Random Forest Regressor : [medium](https://towardsdatascience.com/5-feature-selection-method-from-scikit-learn-you-should-know-ed4d116e4172)
    - WARNING: SUPER LONG TO RUN
  - FDR: sklearn
- Outlier detection: 
  - <span style="color:red"> QUESTION: should we do outlier detection before or after train-test split? </span>
  - <span style="color:red"> QUESTION: are we sure the data to predict doesn't contain outliers? </span>
    
    - [link](https://stats.stackexchange.com/questions/321962/should-i-remove-any-out-liers-before-splitting-the-data)
  - UMAP: TODO
    - Stochastic process: fixing the seed 
    - official [documentation](https://umap-learn.readthedocs.io/en/latest/index.html)
    - Basic parameters for optimization: [Basic UMAP Parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)
  - Isolation Forest: 
    - official [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
  - DBSCAN: 
    - official [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
    - Good article to understand: [Medium](https://medium.com/@dilip.voleti/dbscan-algorithm-for-fraud-detection-outlier-detection-in-a-data-set-60a10ad06ea8)
  - Regressor: 
    - Sklearn StackingRegressor [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
    - Estimators: RidgeCV,LinearSVR,XGBRegressor,AdaBoostRegressor,GaussianRegressor,GradientBoostingRegressor,LGBMRegressor,CatBoostRegressor, ElasticNet, BayesianRidge,KernelRidge (not all converging)
    - GridSearchCV: [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    - BayesSearch [documentation](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html)
    
## TODO
- Maja 
  - [ ] Add a simple explanation in the README and in the docstrings about outlier detection ("outlier score"???)
  - Outlier detection
    - [ ] Make it remove more than 2 outliers
    - [ ] Apply the same method on the TEST set
    - [ ] DBSCAN 
  - [ ] Iterative imputer: save the dataset
- Tristan
  - Feature selection: 
    - [ ] ANOVA 
    - [ ] SFS 
    - [ ] FDR
- Yves
  - [X] GridSearch including preprocessing
  - [ ]	Optimize for meta regressor (the RidgeCV combining the estimators)
  - [ ] Execute extensive GridSearch per regressor >> just extending, narrowing the search space manually since the space is just too big
  - [ ] Find Best combination for Stacking
  - [X] Look into non converging Regressors >> probaly just because of poor parameter choice
     - [X] RidgeCV
     - [X] SVR
     - [X] ElastiNet
  - Clean Code
  - [optional]
  	- Stacking of StackingRegressors >> I belive this is what Team 1 did from last year
  	- BayesSearchCV >> Statistical Hyperparameter tuning
  	
