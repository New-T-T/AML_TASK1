import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def preprocess(df_original: pd.DataFrame, target_original: pd.DataFrame) -> pd.DataFrame:
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
    :return: X_train_preprocess, X_test_preprocess, y_train, y_test
    """
    # Removing the column id as it redundant
    df_original.drop('id', axis=1, inplace=True)

    # train test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(df_original, target_original, test_size=0.2, random_state=42)

    # Imputing missing values with median using sklearn
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

    # Standardizing the features using sklearn
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)
    X_train_standardized = pd.DataFrame(scaler.transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_standardized = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_train_imputed.columns)

    # Removing features with low variance using sklearn
    selector = VarianceThreshold(threshold=0.001)
    selector.fit(X_train_standardized)
    X_train_variance = pd.DataFrame(selector.transform(X_train_standardized), columns=X_train_standardized.columns[selector.get_support()])
    X_test_variance = pd.DataFrame(selector.transform(X_test_standardized), columns=X_train_standardized.columns[selector.get_support()])

    # Removing highly correlated features using sklearn and Pearson correlation
    correlated_features = set()
    X_train_correlation_matrix = X_train_variance.corr()
    for i in range(len(X_train_correlation_matrix.columns)):
        for j in range(i):
            if abs(X_train_correlation_matrix.iloc[i, j]) > 0.9:
                colname = X_train_correlation_matrix.columns[i]
                correlated_features.add(colname)

    X_train_correlation = X_train_variance.drop(labels=correlated_features, axis=1)
    X_test_correlation = X_test_variance.drop(labels=correlated_features, axis=1)

    # Copying the dataframe for export
    X_train_preprocessed = X_train_correlation.copy(deep=True)
    X_test_preprocessed = X_test_correlation.copy(deep=True)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test
