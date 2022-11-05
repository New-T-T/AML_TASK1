from sklearn.ensemble import StackingRegressor



from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV

#define all the estimators/regression models
ESTIMATORS = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42)),
    ('xgb', XGBRegressor()),    #Specify vie XGBREGRESSOR
    ('ada', AdaBoostRegressor()), #Specify via Adaboost regressor
    ('gauss', GaussianProcessRegressor()) #Specify via GAUSSIANPROCESSREGRESSOR
]

#hardcoded past best configurations for the estimators
BEST_ESTIMATORS = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42,epsilon=0.01,C=1)),
    ('xgb', XGBRegressor(learning_rate=0.03,max_depth=4,min_child_weight=2,subsample=0.4,colsample_bytree=0.8,n_estimators=4000,reg_lambda=0.01)),    #Specify vie XGBREGRESSOR
    ('ada', AdaBoostRegressor(n_estimators=2000)), #Specify via Adaboost regressor
    ('gauss', GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())) #Specify via GAUSSIANPROCESSREGRESSOR
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

#define the parameters for grid search
#seperate per regressor
GS_PARAMS_SPLIT = [
    ("lr", {}),
    ("svr", {"epsilon": [0, 0.01, 0.1, 0.3], "C": [1]}),
    ("xgb", {"learning_rate":[0.03],"n_estimators":[4000],"max_depth":[8],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    #("xgb", {"learning_rate":[0.01, 0.03, 0.1],"n_estimators":[100, 2500, 4000, 6000],"max_depth":[4,8,10],"colsample_bytree":[0.8],"reg_lambda": [0.01],"subsample": [0.4],"min_child_weight": [2]}),
    ('ada', {"n_estimators":[100,1000,2000]}),
    ('gauss', {"kernel":[DotProduct() + WhiteKernel()]})
]

BEST_PARAMS = []


#class for out custom stacked regressor
#Seperate class since I am not sure about the implementation of GridSearch on the Stacked regressor, this
# we can write a GridSearch that optimizes each estimator on its own since the parameter space for the whole
# StackedRegressor might be to large
class StackedRegressor():

    #initialized the regrassor given the ESTIMATORS parameters
    def __init__(self):
        self.regressor = StackingRegressor(estimators= BEST_ESTIMATORS , final_estimator=RidgeCV())

    #performs grid search on the parameter space
    """
    def gridsearch(self,x_data,y_data):
        grid = GridSearchCV(estimator=self.regressor, param_grid=GS_PARAMS, cv=5)
        grid.fit(x_data,y_data)
        self.best_params = grid.get_params()
    """
    def gridSearchSeperate(self,x_data,y_data):
        self.best_params =  []
        for i in range(len(ESTIMATORS)):
            grid = GridSearchCV(estimator=ESTIMATORS[i][1], param_grid=GS_PARAMS_SPLIT[i][1], cv=5 , verbose=3)
            grid.fit(x_data,y_data)
            BEST_PARAMS.append(grid.best_params_)
            self.best_params.append(grid.best_params_)
    def from_best_params(self):
        self.regressor = StackingRegressor(estimators = BEST_ESTIMATORS,final_estimator=RidgeCV())

    #fit the regressor by manually handing it the best parameters
    def fit(self,x_data,y_data,params):
        self.regressor.fit(x_data,y_data)

    # fit the regressor by atomatically choosing the best parameters
    # Gridsearch has to be run before this function can be used
    def fit(self,x_data,y_data):
        try:
            self.regressor.fit(x_data,y_data)
        except:
            print("Gridsearch has not yet been performed!")

    def make_prediction(self,x_data):
        return self.regressor.predict(x_data)

    def get_params(self):
        return self.regressor.get_params()
