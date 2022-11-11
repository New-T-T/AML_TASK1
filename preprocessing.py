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

def remove_outliers_from_data(training_set, training_labels, outlier_scores):
    """
    Removes the outliers based on the outlier scores.
    Modifies the scaled training set by removing the outliers from it.
    :param training_set: scaled training set to be modified
    :param training_labels: labels associateed with the training set
    :param outlier_scores: outlier scores computed by a previous algorithm
    :return: modified scaled training set
    """
    X = training_set.copy()
    X_a = training_set[outlier_scores != -1]
    for index in range(0, len(outlier_scores)):
        if outlier_scores[index] == -1:
            training_set = training_set.drop(index=index)
            training_labels = training_labels.drop(index=index)
    # check if X_a and training_set are the same
    print(f'X_a and training_set are the same: {X_a.equals(training_set)}')
    return training_set, training_labels

def remove_outliers(X_train, y_train, outlier_method, umap_enabled: bool, if_contamination: float, if_n_estimators: int,
                    dbs_min_samples: int, dbs_eps: int,
                    seed: int, verbose: int, n_jobs: int):
    """
    Remove outliers from the dataset. Can use IsolationForest or DBSCAN.
    :param X_train: training set
    :param X_test: test set
    :param outlier_method: method to use to remove outliers
    :param umap_enabled: whether to use UMAP or not to reduce the dimensionality of the data
    :param if_contamination: contamination parameter for IsolationForest
    :param if_n_estimators: number of estimators (= trees) for IsolationForest
    :param dbs_min_samples: min_samples parameter for DBSCAN
    :param dbs_eps: Most important DBSCAN parameter. The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param seed: seed for the random number generator
    :param verbose: verbosity level
    :param n_jobs: number of jobs to run in parallel
    :return: X_train, X_test without outliers
    """
    if verbose >= 1:
        print(f"Removing outliers using {outlier_method}")

    if umap_enabled:
        print(f"{'':<1} Using UMAP to reduce dimensionality")
        # Reducing dimensionality with UMAP
        reducer = umap.UMAP(random_state=seed)
        #reducer.fit(X_train)
        embedding = reducer.fit_transform(X_train)

    # Removing outliers with LocalOutlierFactor
    #outlier_scores_lof = sklearn.neighbors.LocalOutlierFactor(contamination=0.001428).fit_predict(embedding.embedding_)
    #remove_outliers(X_training_set, outlier_scores_lof)

    if outlier_method == "IsolationForest":
        if umap_enabled:
            outlier_scores = IsolationForest(n_estimators=if_n_estimators,
                                             contamination=if_contamination,
                                             n_jobs=n_jobs,
                                             random_state=seed).fit_predict(embedding)
        else:
            outlier_scores = IsolationForest(n_estimators=if_n_estimators,
                                             contamination=if_contamination,
                                             n_jobs=n_jobs,
                                             random_state=seed).fit_predict(X_train)
    elif outlier_method == "DBSCAN":
        if umap_enabled:
            outlier_scores = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples,
                                n_jobs=n_jobs).fit_predict(embedding.embedding_)
        else:
            outlier_scores = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples,
                                n_jobs=n_jobs).fit_predict(X_train)

    X_train = X_train[outlier_scores != -1]
    y_train = y_train[outlier_scores != -1]

    #X_train, y_train = remove_outliers_from_data(X_train, y_train, outlier_scores)

    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")
        counter = 0
        for x in outlier_scores:
            if x == -1:
                counter += 1
        print(f"Outliers removed: {counter}")
    return X_train, y_train


def remove_highly_correlated_features(X_train, X_test, threshold: float = 0.9, verbose: int = 1):
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
               umap_enabled: bool,
               if_contamination: float,
               if_n_estimators: int,
               dbs_min_samples: int,
               dbs_eps: int,
               n_jobs: int,
               training: bool, verbose: int, seed: int) -> pd.DataFrame:
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
    :param umap_enabled: whether to use UMAP to reduce dimensionality
    :param if_contamination: contamination parameter for the outlier detection method. Represents the percentage of
                          outliers.
    :param if_n_estimators: number of estimators for the outlier detection method
    :param dbs_min_samples: min_samples parameter for the DBSCAN algorithm
    :param dbs_eps: Most important DBSCAN parameter. The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param verbose: verbose mode, 1 for checkpoints, 2 for dataset shapes
    :param seed: seed for the random number generator
    :param training: whether the data is training data or test data
    :param n_jobs: number of jobs to run in parallel
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
        ('imputation', SimpleImputer(strategy='median')),
        ('scaling', StandardScaler()),
        #('outlier_detection', IsolationForest(n_estimators=1000, contamination=contamination, n_jobs=2, random_state=seed)),
        #('variance_threshold', VarianceThreshold(threshold=0.001))
    ])

    # Imputing and scaling the data
    if verbose >= 1:
        print("Imputing and scaling the training set")
    X_train_index = X_train.copy().index
    X_train = pd.DataFrame(preprocessing_pipeline.fit_transform(X_train, y_train), columns=X_train.columns,
                           index=X_train_index)

    # Removing outliers
    X_train, y_train = remove_outliers(X_train=X_train, y_train=y_train, outlier_method=outlier_method,
                                       umap_enabled=umap_enabled,
                                       if_contamination=if_contamination, if_n_estimators=if_n_estimators,
                                       dbs_min_samples=dbs_min_samples, dbs_eps=dbs_eps,
                                       n_jobs=n_jobs, verbose=verbose, seed=seed)

    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")

    # Removing low variance features
    if verbose >= 1:
        print("Removing low variance features")
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train)
    X_train = pd.DataFrame(selector.transform(X_train), columns=X_train.columns[selector.get_support()],
                           index=X_train.index)
    if verbose >= 2:
        print(f"{'':<1} Shape of the training set: {X_train.shape}")

    # Applying the same preprocessing pipeline to the train_test or test set (depending on the value of training)
    if training:
        X_train_test_index = X_train_test.copy().index
        X_train_test = pd.DataFrame(preprocessing_pipeline.transform(X_train_test), columns=X_train_test.columns,
                                    index=X_train_test_index)
        X_train_test = pd.DataFrame(selector.transform(X_train_test),
                                    columns=X_train_test.columns[selector.get_support()],
                                    index=X_train_test.index)
        # Removing highly correlated features using sklearn and Pearson correlation
        X_train, X_train_test = remove_highly_correlated_features(X_train, X_train_test, verbose=verbose)
        return X_train, X_train_test, y_train, y_train_test

    else:
        X_test_index = X_test.copy().index
        X_test = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns=X_test.columns, index=X_test_index)
        X_test = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()],
                              index=X_test.index)
        # Removing highly correlated features using sklearn and Pearson correlation
        X_train, X_test = remove_highly_correlated_features(X_train, X_test, verbose=verbose)
        return X_train, X_test, y_train
