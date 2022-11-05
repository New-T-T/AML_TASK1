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


    reg = regressor.StackedRegressor()
    #reg.gridSearchSeperate(X_train,y_train.drop(columns=['id']).to_numpy().ravel())
    reg.from_best_params()
    X_total = pd.concat([X_train,X_train_test])
    y_total = pd.concat([y_train,y_train_test])
    print(X_total)
    print(y_total)
    print("Start fitting")
    reg.fit(X_total,y_total.to_numpy().ravel())
    print("Start predicting")
    y_pred_final = reg.make_prediction(X_test)

    ### Output to file
    print("y_pred_final size/shape")
    print(y_pred_final.shape)
    print(y_pred_final.size)
    array_with_id = np.array(np.linspace(1,y_pred_final.size,y_pred_final.size),np.array(y_pred_final))
    output_df = pd.DataFrame(array_with_id, columns=['id','y'])
    output_df.to_csv("output_final.csv")
    output_df = output_df.round(1)
    output_df.to_csv("output_final_rounded.csv")