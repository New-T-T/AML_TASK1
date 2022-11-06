import pandas as pd
from preprocessing import preprocess
from feature_selection import select_features, dummy_rfe_regression, dummy_rfe_regression_predict

if __name__ == '__main__':
    SEED = 40
    training = False
    if training:
        X_train, X_train_test, y_train, y_train_test = preprocess(df_original=pd.read_csv('data/X_train.csv'),
                                                                  target_original=pd.read_csv('data/y_train.csv'),
                                                                  X_test=pd.read_csv('data/X_test.csv'),
                                                                  training=training,
                                                                  outlier_method='IsolationForest',
                                                                  contamination=0.04,
                                                                  dbscan_min_samples=10, # TODO: useless when IsolationForest is used
                                                                  verbose=True, seed=SEED)

        X_train, X_train_test = select_features(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_train_test,
                                                feature_selection_method='FDR',
                                                alpha=0.01,
                                                verbose=True,
                                                timing=True,
                                                )
        dummy_rfe_regression(X_train, y_train, X_train_test, y_train_test, verbose=True, timing=True)
    else:
        X_train, X_test, y_train = preprocess(df_original=pd.read_csv('data/X_train.csv'),
                                              target_original=pd.read_csv('data/y_train.csv'),
                                              X_test=pd.read_csv('data/X_test.csv'),
                                              training=training,
                                              outlier_method='IsolationForest',
                                              contamination=0.04,
                                              dbscan_min_samples=10, # TODO: useless when IsolationForest is used
                                              verbose=True, seed=SEED)
        X_train, X_test = select_features(X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_test,
                                          feature_selection_method='FDR',
                                          alpha=0.01,
                                          verbose=True,
                                          timing=True,
                                          )
        y_predict = dummy_rfe_regression_predict(X_train, y_train, X_test, verbose=True, timing=True)
        y_predict = pd.DataFrame(y_predict)
        y_predict.columns = ['y']
        y_predict.to_csv('data/y_predict.csv', index=True, index_label='id')


