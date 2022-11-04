import pandas as pd

import numpy as np

import regressor
from preprocessing import preprocess


from feature_selection import select_features, dummy_rfe_regression


if __name__ == '__main__':
    df_original = pd.read_csv('data/X_train.csv')
    target_original = pd.read_csv('data/y_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    X_train, X_train_test, y_train, y_train_test, X_test = select_features(df_original=df_original,
                                                                     target_original=target_original,
                                                                     X_test=X_test,
                                                                     alpha=0.3, # 0.4 : 107, 0.3 :
                                                                     verbose=True,
                                                                     timing=True,
                                                                     seed=40)
    dummy_rfe_regression(X_train, y_train, X_train_test, y_train_test, verbose=True, timing=True)

    print(y_train)
    reg = regressor.StackedRegressor()
    #reg.gridSearchSeperate(X_train,y_train.drop(columns=['id']).to_numpy().ravel())
    reg.from_best_params()
    X_total = X_train.join(X_test)
    y_total = y_train.join(y_test)
    reg.fit(X_total,y_total.drop(columns=['id']).to_numpy().ravel())
    y_pred_final = reg.predict(???????????????)

    ### Output to file
    output_df = pd.DataFrame(y_pred_final, columns=['y'])
    output_df.to_csv("output_final.csv")
    output_df = output_df.round(1)
    output_df.to_csv("output_3.csv")