from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor # TO REMOVE when dummy regressor will be removed
from sklearn.feature_selection import SelectFdr, f_regression
import numpy as np
import pandas as pd
import colorama
from colorama import Fore, Style
import time

def select_features(X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    feature_selection_method: str,
                    alpha: float,
                    verbose: int,
                    timing: bool = True) -> pd.DataFrame:

    if verbose >= 1:
        print("Feature selection")

    if timing:
        start_time = time.process_time()

    if feature_selection_method == 'lasso':
        lasso = Lasso(alpha=alpha,
                      fit_intercept=False,
                      max_iter=10000,
                      tol=0.001).fit(X_train, y_train.values.ravel())
        if verbose >= 1:
            if timing:
                print(f"{'':<1} Feature selection time: "
                      f"{Fore.YELLOW}{time.process_time() - start_time:.2f}{Style.RESET_ALL} seconds")
            if verbose >= 2:
                print(f"{'':<1} Lasso picked {colorama.Fore.RED}{sum(lasso.coef_ != 0)}{colorama.Style.RESET_ALL} "
                      f"features and eliminated the other "
                      f"{colorama.Fore.RED}{sum(lasso.coef_ == 0)}{colorama.Style.RESET_ALL} features")

        # Create a mask for the selected features
        mask = lasso.coef_ != 0

    elif feature_selection_method == 'FDR':
        fdr = SelectFdr(f_regression, alpha=alpha).fit(X_train, y_train.values.ravel())
        if verbose:
            if timing:
                print(f"{'':<1} Feature selection time: {Fore.YELLOW}{time.process_time() - start_time:.2f}{Style.RESET_ALL} seconds")
            print(
                f"{'':<1} FDR picked {colorama.Fore.RED}{sum(fdr.get_support())}{colorama.Style.RESET_ALL} features and eliminated the other {colorama.Fore.RED}{sum(fdr.get_support() == False)}{colorama.Style.RESET_ALL} features")

        # Create a mask for the selected features
        mask = fdr.get_support()
        # Apply the mask to the feature dataset

    # Apply the mask to the feature dataset
    X_train = X_train.loc[:, mask]
    X_test = X_test.loc[:, mask]

    return X_train, X_test

def select_features_predict(X_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            feature_selection_method: str,
                            alpha: float,
                            verbose: bool = True,
                            timing: bool = True, seed: int = 40) -> pd.DataFrame:

    if verbose:
        print("Feature selection")
    if timing:
        start_time = time.process_time()

    if feature_selection_method == 'lasso':
        lasso = Lasso(alpha=alpha,
                      fit_intercept=False,
                      max_iter=10000,
                      tol=0.001).fit(X_train, y_train.values.ravel())
        if verbose:
            if timing:
                print(f"{'':<1} Feature selection time: {Fore.YELLOW}{time.process_time() - start_time:.2f}{Style.RESET_ALL} seconds")
            # print(f"{'':<1} Lasso best score: {colorama.Fore.RED}{lasso.score(X_train, y_train)}{colorama.Style.RESET_ALL}")
            # print(f"{'':<1} Lasso best coef: {lasso.coef_}")
            print(
                f"{'':<1} Lasso picked {colorama.Fore.RED}{sum(lasso.coef_ != 0)}{colorama.Style.RESET_ALL} features and eliminated the other {colorama.Fore.RED}{sum(lasso.coef_ == 0)}{colorama.Style.RESET_ALL} features")


        # Create a mask for the selected features
        mask = lasso.coef_ != 0
        # Apply the mask to the feature dataset
        X_train_selected = X_train.loc[:, mask]
        X_test_selected = X_test.loc[:, mask]

    elif feature_selection_method == 'FDR':
        fdr = SelectFdr(f_regression, alpha=alpha).fit(X_train, y_train.values.ravel())
        if verbose:
            if timing:
                print(f"{'':<1} Feature selection time: {Fore.YELLOW}{time.process_time() - start_time:.2f}{Style.RESET_ALL} seconds")
            print(
                f"{'':<1} FDR picked {colorama.Fore.RED}{sum(fdr.get_support())}{colorama.Style.RESET_ALL} features and eliminated the other {colorama.Fore.RED}{sum(fdr.get_support() == False)}{colorama.Style.RESET_ALL} features")

        # Create a mask for the selected features
        mask = fdr.get_support()
        # Apply the mask to the feature dataset
        X_train_selected = X_train.loc[:, mask]
        X_test_selected = X_test.loc[:, mask]
    return X_train_selected, y_train, X_test_selected

def dummy_rfe_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, X_train_test: pd.DataFrame, y_train_test: pd.DataFrame, verbose: bool = True, timing: bool = True) -> pd.DataFrame:
    start_time = time.process_time()
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=0, min_samples_split=2,
                                   min_samples_leaf=1)
    rf_reg.fit(X_train, y_train.values.ravel())
    y_pred = rf_reg.predict(X_train_test)
    if verbose:
        if timing:
            print(
                f"{'':<1} RandomForestRegressor time: {colorama.Fore.YELLOW}{time.process_time() - start_time:.2f}{colorama.Style.RESET_ALL} seconds")
        print(f"{'':<1} RandomForestRegressor r2 score: {colorama.Fore.GREEN}{r2_score(y_train_test, y_pred)}{colorama.Style.RESET_ALL}")
        print(f"{'':<1} RandomForestRegressor mse: {colorama.Fore.GREEN}{mean_squared_error(y_train_test, y_pred)}{colorama.Style.RESET_ALL}")
    return r2_score(y_train_test, y_pred)

def dummy_rfe_regression_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, verbose: bool = True, timing: bool = True) -> pd.DataFrame:
    start_time = time.process_time()
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=0, min_samples_split=2,
                                   min_samples_leaf=1)
    rf_reg.fit(X_train, y_train.values.ravel())
    y_pred = rf_reg.predict(X_test)
    if verbose:
        if timing:
            print(
                f"{'':<1} RandomForestRegressor time: {colorama.Fore.YELLOW}{time.process_time() - start_time:.2f}{colorama.Style.RESET_ALL} seconds")
    return y_pred
