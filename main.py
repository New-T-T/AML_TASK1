import pandas as pd
from preprocessing import preprocess_train, preprocess_predict
from feature_selection import select_features_train, select_features_predict, dummy_rfe_regression, dummy_rfe_regression_predict

if __name__ == '__main__':
    SEED = 40
    training = False
    if training:
        X_train, X_train_test, y_train, y_train_test = preprocess_train(df_original=pd.read_csv('data/X_train.csv'),
                                                                        target_original=pd.read_csv('data/y_train.csv'),
                                                                        outlier_method='IsolationForest',
                                                                        contamination=0.04,
                                                                        dbscan_min_samples=10, # TODO: useless when IsolationForest is used
                                                                        verbose=True, timing=True, seed=SEED)

        X_train, X_train_test, y_train, y_train_test = select_features_train(X_train=X_train,
                                                                             X_train_test=X_train_test,
                                                                             y_train=y_train,
                                                                             y_train_test=y_train_test,
                                                                             feature_selection_method='FDR',
                                                                             alpha=0.01, # 0.4 : 107, 0.3 :
                                                                             verbose=True,
                                                                             timing=True,
                                                                             seed=SEED)
        dummy_rfe_regression(X_train, y_train, X_train_test, y_train_test, verbose=True, timing=True)
    else:
        X_train, y_train, X_test = preprocess_predict(df_original=pd.read_csv('data/X_train.csv'),
                                                      target_original=pd.read_csv('data/y_train.csv'),
                                                      X_test=pd.read_csv('data/X_test.csv'),
                                                      outlier_method='IsolationForest',
                                                      contamination=0.04,
                                                      dbscan_min_samples=10, # TODO: useless when IsolationForest is used
                                                      verbose=True, timing=True, seed=SEED)
        X_train, y_train, X_test = select_features_predict(X_train=X_train,
                                                           y_train=y_train,
                                                           X_test=X_test,
                                                           feature_selection_method='FDR',
                                                           alpha=0.01, # 0.4 : 107, 0.3 :
                                                           verbose=True, timing=True, seed=SEED)
        y_predict = dummy_rfe_regression_predict(X_train, y_train, X_test, verbose=True, timing=True)
        pd.DataFrame(y_predict).to_csv('data/y_predict.csv', index=True)


