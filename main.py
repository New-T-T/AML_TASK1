import pandas as pd
from preprocessing import preprocess


if __name__ == '__main__':
    df_original = pd.read_csv('data/X_train.csv')
    target_original = pd.read_csv('data/y_train.csv')
    X_train, X_test, y_train, y_test = preprocess(df_original=df_original, target_original=target_original)

