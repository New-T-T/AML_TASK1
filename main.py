import pandas as pd
from feature_selection import select_features, dummy_rfe_regression

if __name__ == '__main__':
    df_original = pd.read_csv('data/X_train.csv')
    target_original = pd.read_csv('data/y_train.csv')
    X_train, X_test, y_train, y_test = select_features(df_original=df_original,
                                                       target_original=target_original,
                                                       alpha=0.3, # 0.4 : 107, 0.3 :
                                                       verbose=True,
                                                       timing=True,
                                                       seed=40)
    dummy_rfe_regression(X_train, y_train, X_test, y_test, verbose=True, timing=True)


