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

# warning: weird non-deterministic behaviour
def remove_outliers(training_set, training_labels, outlier_scores):
    """
    Removes the outliers based on the outlier scores.
    Modifies the scaled training set by removing the outliers from it.
    :param training_set: scaled training set to be modified
    :param training_labels: labels associateed with the training set
    :param outlier_scores: outlier scores computed by a previous algorithm
    :return: modified scaled training set
    """

    for index in range(0, len(outlier_scores)):
        if outlier_scores[index] == -1:
            training_set = training_set.drop(index = index)
            training_labels = training_labels.drop(index = index)

    return training_set, training_labels


def preprocess_train(df_original: pd.DataFrame,
                     target_original: pd.DataFrame,
                     outlier_method: str,
                     contamination: float,
                     dbscan_min_samples: int,
                     verbose: bool, timing: bool, seed) -> pd.DataFrame:
    """
    Creates a train and test set from the original data.
    Preprocess the independent features (i.e X) in 4 steps:
    - Median imputation: using SimpleImputer class from sklearn
    - Standardization: using StandardScaler class from sklearn
    - Variance thresholding below 0.001 (0.1%): using VarianceThreshold class from sklearn
    - Correlated features removal: custom function using the correlation matrix (cf: https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/)
    Leave the dependent feature (i.e y) untouched (only split into train and test set).
    :param df_original: original dataframe to preprocess
    :param target_original: original target dataframe to preprocess
    :param outlier_method: method to use to remove outliers
    :param verbose: boolean to print or not the preprocessing steps
    :param timing: boolean to print or not the preprocessing steps timing
    :param seed: seed to use for the random state
    :return: X_train_preprocess, X_test_preprocess, y_train, y_test
    """
    # Making sure parameters are correct
    #assert outlier_method in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope'], "outlier_method must be in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope']"
    assert outlier_method in ['UMAP', 'IsolationForest', 'DBSCAN'], "outlier_method must be in ['UMAP', 'IsolationForest', 'DBSCAN']"

    # Removing the column id as it redundant
    df_original.drop('id', axis=1, inplace=True)
    target_original.drop('id', axis=1, inplace=True)

    # train test split using sklearn
    if verbose:
        print("Train-test split")
    X_train, X_train_test, y_train, y_train_test = train_test_split(df_original, target_original, test_size=0.15, random_state=seed)

    # Imputing missing values with median using sklearn
    if verbose:
        print("Imputing")
    if timing:
        start_imputer = time.process_time()
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_train_test_imputed = pd.DataFrame(imputer.transform(X_train_test), columns=X_train.columns)
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Imputer time: {Fore.YELLOW}{time.process_time() - start_imputer:.2f}{Style.RESET_ALL} seconds")
        print("Scaling")

    """
    start_iterative = time.process_time()
    # Imputing missing values with the sklearn IterativeImputer, for more details see: https://towardsdatascience.com/iterative-imputation-with-scikit-learn-8f3eb22b1a38
    imputer = IterativeImputer(random_state=0, verbose=2)
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_train_test_imputed = pd.DataFrame(imputer.transform(X_train_test), columns=X_train.columns)
    print("IterativeImputer time: " + str(time.process_time() - start_iterative))
    """

    if timing:
        start_scaler = time.process_time()

    # Standardizing the features using sklearn
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    X_train_standardized = pd.DataFrame(scaler.transform(X_train_imputed), columns=X_train_imputed.columns)
    X_train_test_standardized = pd.DataFrame(scaler.transform(X_train_test_imputed), columns=X_train_imputed.columns)

    # # Standardizing the features using sklearn MinMaxScaler
    # scaler = MinMaxScaler()
    # scaler.fit(X_train_imputed)
    # X_train_standardized = pd.DataFrame(scaler.transform(X_train_imputed), columns=X_train_imputed.columns)
    # X_train_test_standardized = pd.DataFrame(scaler.transform(X_train_test_imputed), columns=X_train_imputed.columns)

    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Scaler time: {Fore.YELLOW}{time.process_time() - start_scaler:.2f}{Style.RESET_ALL} seconds")
        print("Outliers detection")

    start_outliers = time.process_time()

    # Outliers detection can be done using UMAP, IsolationForest or DBSCAN
    if outlier_method == 'UMAP':
        if verbose:
            print(f"{'':<1} UMAP")
        if timing:
            start_umap = time.process_time()
        # Reducing dimensionality with UMAP, for more details see: https://arxiv.org/abs/1802.03426
        # Official documentation: https://umap-learn.readthedocs.io/en/latest/index.html
        # Basic parameters: https://umap-learn.readthedocs.io/en/latest/parameters.html
        reducer = umap.UMAP(random_state=seed)  # Fixing the seed according to: https://umap-learn.readthedocs.io/en/latest/reproducibility.html
        embedding = reducer.fit_transform(X_train_standardized)
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} UMAP time: {time.process_time() - start_umap} seconds")

        # Removing outliers with LocalOutlierFactor, for reference see: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
        outlier_scores_lof = LocalOutlierFactor(contamination=0.001428).fit_predict(embedding)
        X_train_standardized, y_train = remove_outliers(X_train_standardized, y_train, outlier_scores_lof)
       # TODO: Do the same for the test set, NOT SURE IF THIS IS CORRECT
        if verbose:
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{len(outlier_scores_lof) - sum(outlier_scores_lof == 1)}{Style.RESET_ALL}")

    elif outlier_method == 'IsolationForest':
        if verbose:
            print(f"{'':<1} IsolationForest")
        if timing:
            start_isolation = time.process_time()
        # Removing outliers with IsolationForest, for reference see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        outlier_scores_isolation = IsolationForest(n_estimators=1000, contamination=contamination, n_jobs=2, random_state=seed).fit_predict(X_train_standardized)
        X_train_standardized = X_train_standardized[outlier_scores_isolation != -1]
        y_train = y_train[outlier_scores_isolation != -1]
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} IsolationForest time: {time.process_time() - start_isolation} seconds")
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{sum(outlier_scores_isolation == -1)}{Style.RESET_ALL}")

    elif outlier_method == 'DBSCAN':
        if verbose:
            print(f"{'':<1} DBSCAN")
        if timing:
            start_dbscan = time.process_time()
        # Removing outliers with DBSCAN, for reference see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        outlier_scores_dbscan = DBSCAN(eps=36, min_samples=dbscan_min_samples, n_jobs=2).fit_predict(X_train_standardized)
        # Dropping the outliers based on the labels of outlier_scores_dbscan
        X_train_standardized = X_train_standardized[outlier_scores_dbscan != -1]
        y_train = y_train[outlier_scores_dbscan != -1]
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} DBSCAN time: {time.process_time() - start_dbscan} seconds")
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{sum(outlier_scores_dbscan == -1)}{Style.RESET_ALL}")


    if verbose:
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_standardized.shape}")
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Outlier detection time: {Fore.YELLOW}{time.process_time() - start_outliers:.2f}{Style.RESET_ALL} seconds")

        print("Variance thresholding")

    """
    # NOTE: We can decide later which strategy works best, so I'm commenting it out for now.
    # Removing outliers with EllipticEnvelope, for more details see: https://towardsdatascience.com/machine-learning-for-anomaly-detection-elliptic-envelope-2c90528df0a6
    outlier_scores_ee = sklearn.covariance.EllipticEnvelope(contamination = 0.1).fit_predict(embedding.embedding_)
    remove_outliers(X_training_set, outlier_scores_ee)
    """

    # Removing features with low variance using sklearn
    if timing:
        start_variance = time.process_time()
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train_standardized)
    X_train_variance = pd.DataFrame(selector.transform(X_train_standardized), columns=X_train_standardized.columns[selector.get_support()])
    X_train_test_variance = pd.DataFrame(selector.transform(X_train_test_standardized), columns=X_train_standardized.columns[selector.get_support()])
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} VarianceThreshold time: {Fore.YELLOW}{time.process_time() - start_variance:.2f}{Style.RESET_ALL} seconds")
        # displaying the number of features removed in green
        print(f"{'':<1} Number of features removed: {Fore.GREEN}{len(X_train_standardized.columns) - len(X_train_variance.columns)}{Style.RESET_ALL}")
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_variance.shape}")
        print("Correlation thresholding")

    # Removing highly correlated features using sklearn and Pearson correlation
    start_correlation = time.process_time()
    correlated_features = set()
    X_train_correlation_matrix = X_train_variance.corr()
    for i in range(len(X_train_correlation_matrix.columns)):
        for j in range(i):
            if abs(X_train_correlation_matrix.iloc[i, j]) > 0.9:
                colname = X_train_correlation_matrix.columns[i]
                correlated_features.add(colname)

    X_train_correlation = X_train_variance.drop(labels=correlated_features, axis=1)
    X_train_test_correlation = X_train_test_variance.drop(labels=correlated_features, axis=1)
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} CorrelationThreshold time: {Fore.YELLOW}{time.process_time() - start_correlation:.2f}{Style.RESET_ALL} seconds")
        # displaying the number of features removed in green
        print(f"{'':<1} Number of features removed: {Fore.GREEN}{len(X_train_variance.columns) - len(X_train_correlation.columns)}{Style.RESET_ALL}")
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_correlation.shape}")

    # Copying the dataframe for export
    X_train_preprocessed = X_train_correlation.copy(deep=True)
    X_train_test_preprocessed = X_train_test_correlation.copy(deep=True)

    return X_train_preprocessed, X_train_test_preprocessed, y_train, y_train_test

def preprocess_predict(df_original: pd.DataFrame,
                       target_original: pd.DataFrame,
                       X_test: pd.DataFrame,
                       outlier_method: str,
                       contamination: float,
                       dbscan_min_samples: int,
                       verbose: bool, timing: bool, seed) -> pd.DataFrame:
    """
    Creates a train and test set from the original data.
    Preprocess the independent features (i.e X) in 4 steps:
    - Median imputation: using SimpleImputer class from sklearn
    - Standardization: using StandardScaler class from sklearn
    - Variance thresholding below 0.001 (0.1%): using VarianceThreshold class from sklearn
    - Correlated features removal: custom function using the correlation matrix (cf: https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/)
    Leave the dependent feature (i.e y) untouched (only split into train and test set).
    :param df_original: original dataframe to preprocess
    :param target_original: original target dataframe to preprocess
    :param X_test: original test dataframe to preprocess
    :param outlier_method: method to use to remove outliers
    :param verbose: boolean to print or not the preprocessing steps
    :param timing: boolean to print or not the preprocessing steps timing
    :param seed: seed to use for the random state
    :return: X_train_preprocess, X_test_preprocess, y_train, y_test
    """
    # Making sure parameters are correct
    #assert outlier_method in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope'], "outlier_method must be in ['None', 'LocalOutlierFactor', 'IsolationForest', 'EllipticEnvelope']"
    assert outlier_method in ['UMAP', 'IsolationForest', 'DBSCAN'], "outlier_method must be in ['UMAP', 'IsolationForest', 'DBSCAN']"

    # Removing the column id as it redundant
    df_original.drop('id', axis=1, inplace=True)
    target_original.drop('id', axis=1, inplace=True)
    X_test.drop('id', axis=1, inplace=True)

    X_train = df_original.copy(deep=True)
    y_train = target_original.copy(deep=True)

    # Imputing missing values with median using sklearn
    if verbose:
        print("Imputing")
    if timing:
        start_imputer = time.process_time()
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Imputer time: {Fore.YELLOW}{time.process_time() - start_imputer:.2f}{Style.RESET_ALL} seconds")
        print("Scaling")

    """
    start_iterative = time.process_time()
    # Imputing missing values with the sklearn IterativeImputer, for more details see: https://towardsdatascience.com/iterative-imputation-with-scikit-learn-8f3eb22b1a38
    imputer = IterativeImputer(random_state=0, verbose=2)
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_train_test_imputed = pd.DataFrame(imputer.transform(X_train_test), columns=X_train.columns)
    print("IterativeImputer time: " + str(time.process_time() - start_iterative))
    """

    if timing:
        start_scaler = time.process_time()

    # Standardizing the features using sklearn
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    X_train_standardized = pd.DataFrame(scaler.transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_standardized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # # Standardizing the features using sklearn MinMaxScaler
    # scaler = MinMaxScaler()
    # scaler.fit(X_train_imputed)
    # X_train_standardized = pd.DataFrame(scaler.transform(X_train_imputed), columns=X_train_imputed.columns)
    # X_train_test_standardized = pd.DataFrame(scaler.transform(X_train_test_imputed), columns=X_train_imputed.columns)

    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Scaler time: {Fore.YELLOW}{time.process_time() - start_scaler:.2f}{Style.RESET_ALL} seconds")
        print("Outliers detection")

    start_outliers = time.process_time()

    # Outliers detection can be done using UMAP, IsolationForest or DBSCAN
    if outlier_method == 'UMAP':
        if verbose:
            print(f"{'':<1} UMAP")
        if timing:
            start_umap = time.process_time()
        # Reducing dimensionality with UMAP, for more details see: https://arxiv.org/abs/1802.03426
        # Official documentation: https://umap-learn.readthedocs.io/en/latest/index.html
        # Basic parameters: https://umap-learn.readthedocs.io/en/latest/parameters.html
        reducer = umap.UMAP(random_state=seed)  # Fixing the seed according to: https://umap-learn.readthedocs.io/en/latest/reproducibility.html
        embedding = reducer.fit_transform(X_train_standardized)
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} UMAP time: {time.process_time() - start_umap} seconds")

        # Removing outliers with LocalOutlierFactor, for reference see: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
        outlier_scores_lof = LocalOutlierFactor(contamination=0.001428).fit_predict(embedding)
        X_train_standardized, y_train = remove_outliers(X_train_standardized, y_train, outlier_scores_lof)
       # TODO: Do the same for the test set, NOT SURE IF THIS IS CORRECT
        if verbose:
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{len(outlier_scores_lof) - sum(outlier_scores_lof == 1)}{Style.RESET_ALL}")

    elif outlier_method == 'IsolationForest':
        if verbose:
            print(f"{'':<1} IsolationForest")
        if timing:
            start_isolation = time.process_time()
        # Removing outliers with IsolationForest, for reference see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        outlier_scores_isolation = IsolationForest(n_estimators=1000, contamination=contamination, n_jobs=2, random_state=seed).fit_predict(X_train_standardized)
        X_train_standardized = X_train_standardized[outlier_scores_isolation != -1]
        y_train = y_train[outlier_scores_isolation != -1]
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} IsolationForest time: {time.process_time() - start_isolation} seconds")
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{sum(outlier_scores_isolation == -1)}{Style.RESET_ALL}")

    elif outlier_method == 'DBSCAN':
        if verbose:
            print(f"{'':<1} DBSCAN")
        if timing:
            start_dbscan = time.process_time()
        # Removing outliers with DBSCAN, for reference see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        outlier_scores_dbscan = DBSCAN(eps=36, min_samples=dbscan_min_samples, n_jobs=2).fit_predict(X_train_standardized)
        # Dropping the outliers based on the labels of outlier_scores_dbscan
        X_train_standardized = X_train_standardized[outlier_scores_dbscan != -1]
        y_train = y_train[outlier_scores_dbscan != -1]
        if verbose:
            if timing:
                # fstring
                print(f"{'':<1} DBSCAN time: {time.process_time() - start_dbscan} seconds")
            # displaying the number of outliers removed in green
            print(f"{'':<1} Number of outliers removed: {Fore.GREEN}{sum(outlier_scores_dbscan == -1)}{Style.RESET_ALL}")


    if verbose:
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_standardized.shape}")
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} Outlier detection time: {Fore.YELLOW}{time.process_time() - start_outliers:.2f}{Style.RESET_ALL} seconds")

        print("Variance thresholding")

    """
    # NOTE: We can decide later which strategy works best, so I'm commenting it out for now.
    # Removing outliers with EllipticEnvelope, for more details see: https://towardsdatascience.com/machine-learning-for-anomaly-detection-elliptic-envelope-2c90528df0a6
    outlier_scores_ee = sklearn.covariance.EllipticEnvelope(contamination = 0.1).fit_predict(embedding.embedding_)
    remove_outliers(X_training_set, outlier_scores_ee)
    """

    # Removing features with low variance using sklearn
    if timing:
        start_variance = time.process_time()
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train_standardized)
    X_train_variance = pd.DataFrame(selector.transform(X_train_standardized), columns=X_train_standardized.columns[selector.get_support()])
    X_test_variance = pd.DataFrame(selector.transform(X_test_standardized), columns=X_test_standardized.columns[selector.get_support()])
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} VarianceThreshold time: {Fore.YELLOW}{time.process_time() - start_variance:.2f}{Style.RESET_ALL} seconds")
        # displaying the number of features removed in green
        print(f"{'':<1} Number of features removed: {Fore.GREEN}{len(X_train_standardized.columns) - len(X_train_variance.columns)}{Style.RESET_ALL}")
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_variance.shape}")
        print("Correlation thresholding")

    # Removing highly correlated features using sklearn and Pearson correlation
    start_correlation = time.process_time()
    correlated_features = set()
    X_train_correlation_matrix = X_train_variance.corr()
    for i in range(len(X_train_correlation_matrix.columns)):
        for j in range(i):
            if abs(X_train_correlation_matrix.iloc[i, j]) > 0.9:
                colname = X_train_correlation_matrix.columns[i]
                correlated_features.add(colname)

    X_train_correlation = X_train_variance.drop(labels=correlated_features, axis=1)
    X_test_correlation = X_test_variance.drop(labels=correlated_features, axis=1)
    if verbose:
        if timing:
            # display the only the time in yellow
            print(f"{'':<1} CorrelationThreshold time: {Fore.YELLOW}{time.process_time() - start_correlation:.2f}{Style.RESET_ALL} seconds")
        # displaying the number of features removed in green
        print(f"{'':<1} Number of features removed: {Fore.GREEN}{len(X_train_variance.columns) - len(X_train_correlation.columns)}{Style.RESET_ALL}")
        # displaying the shape of the training set
        print(f"{'':<1} Shape of the training set: {X_train_correlation.shape}")

    # Copying the dataframe for export
    X_train_preprocessed = X_train_correlation.copy(deep=True)
    X_test_preprocessed = X_test_correlation.copy(deep=True)

    return X_train_preprocessed, y_train, X_test_preprocessed
