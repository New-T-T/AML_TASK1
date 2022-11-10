import pandas as pd

import regressor
from preprocessing import preprocess
from feature_selection import select_features, dummy_rfe_regression, dummy_rfe_regression_predict

from sklearn.metrics import r2_score

if __name__ == '__main__':
    SEED = 40
    NJOBS = 2
    training = True
    if training:
        X_train, X_train_test, y_train, y_train_test = preprocess(df_original=pd.read_csv('data/X_train.csv'),
                                                                  target_original=pd.read_csv('data/y_train.csv'),
                                                                  X_test=pd.read_csv('data/X_test.csv'),
                                                                  training=training,
                                                                  outlier_method='IsolationForest',  # 'DBSCAN' or 'IsolationForest'
                                                                  umap_enabled=False,  # [True, False]
                                                                  if_contamination=0.04,  # contamination parameter for IsolationForest, percentage of outliers in the dataset: [0.04, 0.06, 0.08, 0.1]
                                                                  if_n_estimators=1000,  # number of estimators for IsolationForest [100, 500, 1000, 2000]
                                                                  dbs_min_samples=10, # [5, 10, 15, 20]
                                                                  dbs_eps=36, # default 0.5, so I don't know what to put here
                                                                  n_jobs=NJOBS,
                                                                  verbose=2, seed=SEED)

        X_train, X_train_test = select_features(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_train_test,
                                                feature_selection_method='FDR', # ['FDR', 'lasso']
                                                alpha=0.01, # FDR and Lasso, higher = less features selected. [0.01, 0.05, 0.1, 0.2, 0.5]
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
                                              umap_enabled=True,
                                              if_contamination=0.04,
                                              if_n_estimators=1000,
                                              dbs_min_samples=10,
                                              dbs_eps=36,
                                              n_jobs=NJOBS,
                                              verbose=2, seed=SEED)
        X_train, X_test = select_features(X_train=X_train,
                                          y_train=y_train,
                                          X_test=X_test,
                                          feature_selection_method='FDR',
                                          alpha=0.01,
                                          verbose=True,
                                          timing=True,
                                          )

        reg = regressor.StackedRegressor()
        reg.gridsearchseperate(X_train, y_train.values.ravel())
        #reg.from_best_params()
        reg.fit(X_train, y_train.values.ravel())
        y_predict = reg.make_prediction(X_test)



        y_predict = pd.DataFrame(y_predict)
        y_predict.columns = ['y']
        y_predict.to_csv('data/y_predict.csv', index=True, index_label='id')
