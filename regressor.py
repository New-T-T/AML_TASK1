from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV
#from scikit.skopt import BayesSearchCV


from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge



#define all the estimators/regression models
#these need to be commented out in the same pattern as in the gridsearch parameter definition since the names are accesed here
ESTIMATORS = [
    #('lr', RidgeCV()),                                     #not converging
    #('svr', LinearSVR(random_state=42)),                   #not converging  # not linearly seperable
    ('xgb', XGBRegressor()),
    ('ada', AdaBoostRegressor()),
    #('gauss', GaussianProcessRegressor()),
    ('gradient_regressor', GradientBoostingRegressor()),
    ('lgbm', LGBMRegressor()),
    ('cat', CatBoostRegressor()),
    #('elastic_net', ElasticNet()),                          #not converging
    #('bay_ridge', BayesianRidge()),
    #('kernel_ridge', KernelRidge())                          #not converging
]

#hardcoded past best configurations for the estimators
BEST_ESTIMATORS = [
    #('lr', RidgeCV()),
    #('svr', LinearSVR(random_state=42,epsilon=0.01,C=1)),
    ('xgb', XGBRegressor(learning_rate=0.03,max_depth=4,min_child_weight=2,subsample=0.4,colsample_bytree=0.8,n_estimators=4000,reg_lambda=0.01)),
    ('ada', AdaBoostRegressor(n_estimators=1000,learning_rate=1)),
    ('gauss', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())),
    ('gradient_regressor', GradientBoostingRegressor(n_estimators=100)),
    ('lgbm', LGBMRegressor(boosting_type='gbdt',colsample_bytree=0.65, n_estimators= 24, num_leaves=16, reg_alpha= 1.2, reg_lambda= 1.2, subsample= 0.7)),
    ('cat,', CatBoostRegressor(depth=6,learning_rate=0.1,l2_leaf_reg=7)),
    #('elastic_net', ElasticNet(l1_ratio=0)),
    ('bay_ridge', BayesianRidge(alpha_1=1e-07, alpha_2= 1e-05, lambda_1= 1e-05, lambda_2= 1e-07, n_iter= 300)),
    #('kernel_ridge',  KernelRidge())
]

# define the parameter space for the grid search
#over the stacked regressor
GS_PARAMS = {
    "svr__epsilon": [0, 0.01, 0.1, 0.3],
    "svr__C": [10],
    "xgb__learning_rate":[0.01, 0.03, 0.1],
    "xgb__n_estimators":[100, 2500, 4000, 6000],
    "xgb__max_depth":[4,8,10],
    "xgb__colsample_bytree":[0.8],
    "xgb__reg_lambda": [0.01],
    "xgb__subsample": [0.4],
    "xgb__min_child_weight": [2]

    #ADD FURTHER PARAMETERS FOR SEARCH
    #GET NAMING SCHEME FROM regressor.get_params()
}

#define the parameters for gridsearchseperate
#seperate per regressor
GS_PARAMS_SPLIT_TEST = [
    #("lr", {}),
    #("svr", {"epsilon": [0, 0.01, 0.1, 0.3], "C": [1]}),
    ("xgb", {"learning_rate": [ 0.03],"n_estimators":[ 4000],"max_depth":[8],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    ('ada', {"n_estimators": [1000],"learning_rate":[1]}),
    #('gauss', {"kernel":[DotProduct() + WhiteKernel()]}),
    ('gradient_regressor', {"n_estimators":[100]}),
    ('lgbm', { 'n_estimators': [24],'num_leaves': [16], 'boosting_type' : ['gbdt'],'colsample_bytree' : [ 0.65],'subsample' : [0.7],'reg_alpha' : [1.2],'reg_lambda' : [1.2]}),
    ('cat', {'learning_rate': [ 0.1], 'depth': [ 6], 'l2_leaf_reg': [3, 5, 7]}),
    #('elastic_net', {'l1_ratio': [0,1,30]}),
    #('bay_ridge', {'n_iter':[300], 'alpha_1':[1e-07],'alpha_2':[1e-05],'lambda_1':[1e-05],'lambda_2':[1e-07]}),
    #('kernel_ridge', {})
]
#count numbers of configurations, since seperate is enabled this is the amout of times the GridSearchCV will run per preprocessing configuration
NUMBER_CONFIG_REGRESSORS = (3*3*3) + (3*3) + (1) + (2) +(3 * 4) + (3*3*3) +(2*2*2*2)
GS_PARAMS_SPLIT = [
    #("lr", {}),
    #("svr", {"epsilon": [0, 0.01, 0.1, 0.3], "C": [1]}),
    ("xgb", {"learning_rate": [ 0.03,0.1],"n_estimators":[2500, 4000, 6000],"max_depth":[4,8,10],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    ('ada', {"n_estimators": [100,1000,2000],"learning_rate":[0.1,1,10]}),
    ('gauss', {"kernel":[DotProduct() + WhiteKernel()]}),
    ('gradient_regressor', {"n_estimators":[10,100]}),
    ('lgbm', { 'n_estimators': [8,16,24],'num_leaves': [6,8,12,16], 'boosting_type' : ['gbdt', 'dart'],'colsample_bytree' : [ 0.65],'subsample' : [0.7],'reg_alpha' : [1.2],'reg_lambda' : [1.2]}),
    ('cat', {'learning_rate': [0.03, 0.1], 'depth': [4, 6, 10], 'l2_leaf_reg': [3, 5, 7]}),
    #('elastic_net', {'l1_ratio': [0,1,30]}),
    ('bay_ridge', {'n_iter':[300,600], 'alpha_1':[1e-07],'alpha_2':[1e-06,1e-05],'lambda_1':[1e-06,1e-05],'lambda_2':[1e-07,1e-06]}),
    #('kernel_ridge', {})
]

#For BayesSearch not yet implemented
GS_PARAMS_SPLIT_BAYES = [
    #("lr", {}),
    #("svr", {"epsilon": [0, 0.01, 0.1, 0.3], "C": [1]}),
    #("xgb", {"learning_rate":[0.03],"n_estimators":[4000],"max_depth":[8],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    #("xgb", {"learning_rate":[0.01, 0.03, 0.1],"n_estimators":[100, 2500, 4000, 6000],"max_depth":[4,8,10],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    #('ada', {"n_estimators":[100,1000,2000],"learning_rate":[0.1,1,10]}),
    #('gauss', {"kernel":[DotProduct() + WhiteKernel()]}),
    #('gradient_regressor', {"n_estimators":[10,100]}),
    ('lgbm', { 'n_estimators': [8,16,24],'num_leaves': [6,8,12,16], 'boosting_type' : ['gbdt', 'dart'],'colsample_bytree' : [ 0.65],'subsample' : [0.7],'reg_alpha' : [1.2],'reg_lambda' : [1.2]}),
    #('cat', {'learning_rate': [0.03, 0.1], 'depth': [4, 6, 10], 'l2_leaf_reg': [3, 5, 7]}),
    ('elastic_net', {'l1_ratio': [0,1,30]}),
    ('bay_ridge', {})
]


def evaluate_estimator(estimator, X_train, X_validation , y_train, y_validation):
    estimator.fit(X_train, y_train)
    return r2_score(y_validation, estimator.predict(X_validation))


#class for out custom stacked regressor
#Seperate class since I am not sure about the implementation of GridSearch on the Stacked regressor, this
# we can write a GridSearch that optimizes each estimator on its own since the parameter space for the whole
# StackedRegressor might be to large
class StackedRegressor():

    #initialized the regressor given the ESTIMATORS parameters
    def __init__(self):
        self.regressor = StackingRegressor(estimators=BEST_ESTIMATORS , final_estimator=RidgeCV(),verbose=3)
        self.best_params = []
        self.number_config = 1

    #performs grid search on the parameter space
    #this is the whole parameter space, takes too long
    def gridsearch(self,x_data,y_data):
        grid = GridSearchCV(estimator=self.regressor, param_grid=GS_PARAMS, cv=5)
        grid.fit(x_data,y_data)
        self.best_params = grid.get_params()

    def gridsearchseperate(self,x_data,y_data):
        reg_list = []
        for i in range(len(ESTIMATORS)):
            grid = GridSearchCV(estimator=ESTIMATORS[i][1], param_grid=GS_PARAMS_SPLIT_TEST[i][1], cv=5 , verbose=3)
            grid.fit(x_data, y_data)
            self.best_params.append(grid.best_params_)
            reg_list.append((ESTIMATORS[i][0], grid.best_estimator_))
        combined_regressor = StackingRegressor(estimators=reg_list , final_estimator=RidgeCV(),verbose=3)
        return combined_regressor


    def gridsearchseperate_bayes(self,x_data,y_data):
        #for i in range(len(ESTIMATORS)):
           # grid = BayesSearchCV(estimator=ESTIMATORS[i][1], param_grid=GS_PARAMS_SPLIT_BAYES[i][1], cv=5 , verbose=3)
            #grid.fit(x_data,y_data)
            #BEST_PARAMS.append(grid.best_params_)
            #self.best_params.append(grid.best_params_)

        file1 = open("data/best_params_bayes.txt", "w")  # write mode
        file1.write(str(GS_PARAMS_SPLIT))
        file1.write(str(self.best_params))
        file1.close()


    def from_best_params(self):
        self.regressor = StackingRegressor(estimators=BEST_ESTIMATORS,final_estimator=RidgeCV())

    #fit the regressor by manually handing it the best parameters
    def fit(self,x_data,y_data):
        self.regressor.fit(x_data,y_data)

    def predict(self,x_data):
        return self.regressor.predict(x_data)

    def get_params(self):
        return self.regressor.get_params()
