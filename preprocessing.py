import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import umap
from colorama import Fore, Style
from sklearn.cluster import DBSCAN
from imblearn.pipeline import Pipeline
import pickle

def remove_outliers(X_train, y_train, outlier_method, contamination: float, dbscan_min_samples: int, seed: int, verbose: int):
    """
    Remove outliers from the dataset. Can use IsolationForest or DBSCAN.
    :param X_train: training set
    :param X_test: test set
    :param outlier_method: method to use to remove outliers
    """
    # Reducing dimensionality with UMAP
    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(X_train)

    if verbose >= 1:
        print(f"Removing outliers using {outlier_method}")
    if outlier_method == "IsolationForest":
        outlier_scores = IsolationForest(n_estimators=1000,
                                         contamination=contamination,
                                         n_jobs=2,
                                         random_state=seed).fit_predict(embedding)
    elif outlier_method == "DBSCAN":
        outlier_scores = DBSCAN(eps=36, min_samples=dbscan_min_samples, n_jobs=2).fit_predict(embedding)

    X_train = X_train[outlier_scores != -1]
    y_train = y_train[outlier_scores != -1]

    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")
        counter = 0
        for x in outlier_scores:
            if x == -1:
                counter += 1
        print(f"Outliers removed: {counter}")
    return X_train, y_train


def remove_highly_correlated_features(X_train, X_test, threshold: float = 0.9, verbose: bool = False):
    """
    Remove highly correlated features from the dataset.
    :param X_train: training set
    :param X_test: test set
    :param threshold: threshold for the correlation between features above which the features are removed
    :param verbose: verbosity level
    :return: X_train, X_test without highly correlated features
    """
    if verbose >= 1:
        print(f"Removing highly correlated features")
    correlated_features = set()
    X_train_correlation_matrix = X_train.corr()
    for i in range(len(X_train_correlation_matrix.columns)):
        for j in range(i):
            if abs(X_train_correlation_matrix.iloc[i, j]) > threshold:
                colname = X_train_correlation_matrix.columns[i]
                correlated_features.add(colname)

    X_train = X_train.drop(labels=correlated_features, axis=1)
    X_test = X_test.drop(labels=correlated_features, axis=1)
    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")
    return X_train, X_test


def preprocess(df_original: pd.DataFrame,
               target_original: pd.DataFrame,
               X_test: pd.DataFrame,
               outlier_method: str,
               contamination: float,
               dbscan_min_samples: int,
               training: bool, verbose: bool, seed: int) -> pd.DataFrame:
    """
    Preprocesses the data. The pipeline is as follows:
    0. Train/test split (if training is True)
    1. Impute missing values
    2. Scale the data
    3. Remove outliers (can be done using IsolationForest or DBSCAN)
    4. Remove low variance features
    5. Remove highly correlated features

    :param df_original: original dataframe
    :param target_original: original target dataframe
    :param outlier_method: outlier detection method
    :param contamination: contamination parameter for the outlier detection method. Represents the percentage of
                          outliers.
    :param dbscan_min_samples: min_samples parameter for the DBSCAN algorithm
    :param verbose: verbose mode, 1 for checkpoints, 2 for dataset shapes
    :param seed: seed for the random number generator
    :return: preprocessed dataframe
    """
    # Making sure parameters are correct
    #assert outlier_method in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope'], "outlier_method must be in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope']"
    assert outlier_method in ['UMAP', 'IsolationForest', 'DBSCAN'], "outlier_method must be in ['UMAP', 'IsolationForest', 'DBSCAN']"

    # Removing the column id as it redundant
    df_original.drop('id', axis=1, inplace=True)
    target_original.drop('id', axis=1, inplace=True)
    X_test.drop('id', axis=1, inplace=True)
    X_train = None
    y_train = None
    X_train_test = None
    y_train_test = None

    if training:
        # Splitting the data into training and test sets
        print(f"Splitting the data into training and test sets (85/15)")
        X_train, X_train_test, y_train, y_train_test = train_test_split(df_original, target_original, test_size=0.15,
                                                                        random_state=seed)
    else:
        print(f"NO SPLITTING of the data into training and test sets")
        X_train = df_original
        y_train = target_original

    # Creating a (simple, for now) preprocessing pipeline
    # 1. Imputation
    # 2. Scaling
    preprocessing_pipeline = Pipeline(steps=[
        #('imputation', SimpleImputer(strategy='median')),
        ('scaling', StandardScaler()),
        #('outlier_detection', IsolationForest(n_estimators=1000, contamination=contamination, n_jobs=2, random_state=seed)),
        #('variance_threshold', VarianceThreshold(threshold=0.001))
    ])

    # Imputing and scaling the data
    if verbose >= 1:
        print("Imputing and scaling the training set")
    X_train = pd.read_pickle('imputer.pkl', compression='gzip')
    X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train, y_train), columns=X_train.columns)

    # Removing outliers
    X_train, y_train = remove_outliers(X_train, y_train, outlier_method, contamination, dbscan_min_samples, seed, verbose)
    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")

    # Removing low variance features
    if verbose >= 1:
        print("Removing low variance features")
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train)
    X_train = pd.DataFrame(selector.transform(X_train), columns=X_train.columns[selector.get_support()])
    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")

    # Applying the same preprocessing pipeline to the train_test or test set (depending on the value of training)
    if training:
        X_train_test = pd.DataFrame(preprocessing_pipeline.transform(X_train_test), columns=X_train_test.columns)
        X_train_test = pd.DataFrame(selector.transform(X_train_test),
                                    columns=X_train_test.columns[selector.get_support()])
        # Removing highly correlated features using sklearn and Pearson correlation
        X_train, X_train_test = remove_highly_correlated_features(X_train, X_train_test, verbose=verbose)
        return X_train, X_train_test, y_train, y_train_test

    else:
        X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns=X_test.columns)
        X_test = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()])
        # Removing highly correlated features using sklearn and Pearson correlation
        X_train, X_test = remove_highly_correlated_features(X_train, X_test, verbose=verbose)
        return X_train, X_test, y_train
