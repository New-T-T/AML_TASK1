import pandas as pd
from feature_selection import select_features, dummy_rfe_regression

if __name__ == '__main__':
    df_original = pd.read_csv('data/X_train.csv')
    target_original = pd.read_csv('data/y_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    X_train, X_train_test, y_train, y_train_test, X_test = select_features(df_original=df_original,
                                                                           target_original=target_original,
                                                                           X_test=X_test,
                                                                           outlier_method='IsolationForest',
                                                                           alpha=0.5, # 0.4 : 107, 0.3 :
                                                                           verbose=True,
                                                                           timing=True,
                                                                           seed=40)
    dummy_rfe_regression(X_train, y_train, X_train_test, y_train_test, verbose=True, timing=True)


