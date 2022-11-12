from feature_selection import select_features
import regressor
from preprocessing import preprocess
import pandas as pd
from sklearn.metrics import r2_score

from sklearn import pipeline


#defines the gridsearch parameters

#preprocess parameters
GRID_OUTLIER_METHODS = ["IsolationForest","DBSCAN"]
GRID_FEATURE_SELECTION_METHODS = ["lasso","FDR"]
EPS_GRID = [36, 60]     #dont go lower it will remove everything
MIN_SAMPLES_GRID = [15, 20]  #dont go lower it will remove everything
II_ENABLED_GRID = [ False]
N_ESTIMATORS_GRID = [ 500, 1000, 2000]
ALPHA_GRID = [0.05, 0.1]
UMAP_GRID = [False,True]
FR_ET_N_GRID = [100,1000]

#feature selection parameters
FR_ALPHA_GRID = [0.01, 0.1, 0.5]  # Yves 0.01 # Maya 0.1 # Tristan 0.5
FR_METHOD_GRID = ['FDR', 'lasso']

#calculate total amount of configurations in preprocessing space
NUMBER_CONFIGS_TOP = len(MIN_SAMPLES_GRID) * len(EPS_GRID) * len(FR_METHOD_GRID) * len(FR_ALPHA_GRID)* len(UMAP_GRID) * len(II_ENABLED_GRID) * len(FR_ET_N_GRID)
NUMBER_CONFIGS_BOTTOM = len(ALPHA_GRID) * len(N_ESTIMATORS_GRID) * len(FR_METHOD_GRID) * len(FR_ALPHA_GRID)* len(UMAP_GRID) * len(II_ENABLED_GRID)* len(FR_ET_N_GRID)
TOTAL_CONFIGS_PRE = NUMBER_CONFIGS_TOP + NUMBER_CONFIGS_BOTTOM

SEED = 40
NJOBS = 5


class custom_Gridsearch():

    def __init__(self):
        self.reg = regressor.StackedRegressor()
        self.curr_score = 0
        self.best_params_pre = {}
        self.regressor_scores = {}
        self.best_model = regressor.StackedRegressor()

    def perform_gridsearch(self):
        self.perform_search_outlier()

    def perform_search_outlier(self):
        self.counter = 0
        self.best_params = []
        for umap_flag in UMAP_GRID:
            for iter_enabled in II_ENABLED_GRID:
                for method in GRID_OUTLIER_METHODS:
                    if method == "DBSCAN":
                        for eps in EPS_GRID: # 0.5= Yves 10 = Maya 36=Tristan
                            for min_samples in MIN_SAMPLES_GRID:
                                if umap_flag:
                                    print("skipped due to umap")
                                else:
                                    X_train, X_train_test, y_train, y_train_test = preprocess(
                                                                                            ii_enabled = iter_enabled,
                                                                                            df_original=pd.read_csv('data/X_train.csv'),
                                                                                            target_original=pd.read_csv('data/y_train.csv'),
                                                                                            X_test=pd.read_csv('data/X_test.csv'),
                                                                                            training=True,
                                                                                            outlier_method='DBSCAN',  # 'DBSCAN' or 'IsolationForest'
                                                                                            umap_enabled=False,  # [True, False]
                                                                                            if_contamination=0.04,
                                                                                            # contamination parameter for IsolationForest, percentage of outliers in the dataset: [0.04, 0.06, 0.08, 0.1]
                                                                                            if_n_estimators=1000,  # number of estimators for IsolationForest [100, 500, 1000, 2000]
                                                                                            dbs_min_samples=min_samples,  # [5, 10, 15, 20]
                                                                                            dbs_eps=eps,  # default 0.5, so I don't know what to put here
                                                                                            n_jobs=NJOBS,
                                                                                         verbose=2, seed=SEED)
                                    param_dict = {
                                        ('umap_flag', umap_flag),
                                        ('outlier_method', "DBSCAN"),
                                        ('dbs_eps', eps),
                                        ('dbs_min_samples', min_samples)
                                    }
                                    self.perform_gridsearch_select_feature(X_train, X_train_test, y_train, y_train_test, param_dict)


                    elif method == "IsolationForrest" :
                        for n_estimators in  N_ESTIMATORS_GRID:
                            for contamination in ALPHA_GRID:
                                X_train, X_train_test, y_train, y_train_test = preprocess(
                                                                                            ii_enabled = iter_enabled,
                                                                                            df_original=pd.read_csv('data/X_train.csv'),
                                                                                            target_original=pd.read_csv('data/y_train.csv'),
                                                                                            X_test=pd.read_csv('data/X_test.csv'),
                                                                                            training=True,
                                                                                            outlier_method='IsolationForest',  # 'DBSCAN' or 'IsolationForest'
                                                                                            umap_enabled=umap_flag,  # [True, False]
                                                                                            if_contamination=contamination,
                                                                                            # contamination parameter for IsolationForest, percentage of outliers in the dataset: [0.04, 0.06, 0.08, 0.1]
                                                                                            if_n_estimators=n_estimators,  # number of estimators for IsolationForest [100, 500, 1000, 2000]
                                                                                            dbs_min_samples=10,  # [5, 10, 15, 20]
                                                                                            dbs_eps=36,  # default 0.5, so I don't know what to put here
                                                                                            n_jobs=NJOBS,
                                                                                            verbose=2, seed=SEED)

                                param_dict = {
                                    ('umap_flag', umap_flag),
                                    ('outlier_method', "IsolationForrest"),
                                    ('if_n_estimators', n_estimators),
                                    ('if_contamination', contamination)
                                }
                                self.perform_gridsearch_select_feature(X_train, X_train_test, y_train, y_train_test,param_dict)


    def perform_gridsearch_select_feature(self,X_train, X_train_test, y_train, y_train_test,param_dict):
        for method in FR_METHOD_GRID:
            for alpha in FR_ALPHA_GRID:
                for et_n in FR_ET_N_GRID:
                    X_train, X_train_test = select_features(X_train=X_train,
                                                            y_train=y_train,
                                                            X_test=X_train_test,
                                                            feature_selection_method=method,  # ['FDR', 'lasso']
                                                            alpha=alpha,
                                                            # FDR and Lasso, higher = less features selected. [0.01, 0.05, 0.1, 0.2, 0.5]
                                                            verbose=True,
                                                            timing=True,
                                                            et_n_best=et_n,
                                                            )

                    #perform gridsearch on the regressor
                    reg = regressor.StackedRegressor()
                    best_reg = reg.gridsearchseperate(X_train,y_train)
                    best_reg.fit(X_train,y_train)
                    score = r2_score(y_train_test, best_reg.predict(X_train_test))
                    self.counter = self.counter + 1

                    #if score is better than one seen so far print configuration into file and save score
                    #this is done every time one is found to keep a log in case of crashes or errors
                    if score > self.curr_score:
                        self.curr_score = score
                        self.best_model = best_reg.named_estimators_
                        param_dict.update({method:alpha})
                        for name,estimator in best_reg.named_estimators_.items():
                            self.regressor_scores.update({name: regressor.evaluate_estimator(estimator, X_train, X_train_test, y_train, y_train_test)})
                        """
                        for i in range(len(best_reg.estimators_)):
                            self.regressor_scores.update({best_reg.named_estimators_[i][0]:regressor.evaluate_estimator(best_reg.estimators_[i],X_train, X_train_test, y_train, y_train_test)})
                        self.best_params_pre = param_dict
                        """
                        file1 = open("data/best_params_custom_Yves.txt", "a")  # write mode
                        file1.write("Config Start \n")
                        file1.write("Curr_score:" + str(self.curr_score) + "\n")
                        file1.write("Best_params_pre:" + str(param_dict) + "\n")
                        file1.write("Best_model:" + str(self.best_model) + "\n")
                        file1.write("Regressor scores:" + str(self.regressor_scores) + "\n")
                        file1.write("Config End \n")
                        file1.close()

                    #write progress of gridsearch into file
                    #has to be done to a file since because there is too much console output, and keeps track of
                    file1 = open("data/best_params_custom_Yves.txt", "a")  # write mode
                    file1.write("Executed configurations: " + str(self.counter) + "/" + str(TOTAL_CONFIGS_PRE * reg.number_config) + "\n")
                    file1.close()
