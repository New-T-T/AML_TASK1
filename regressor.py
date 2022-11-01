from sklearn.ensemble import StackingRegressor



from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

#define all the estimators/regression models
ESTIMATORS = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42)),
    ('xgb', XGBRegressor()),    #Specify vie XGBREGRESSOR
    ('ada', AdaBoostRegressor()), #Specify via Adaboost regressor
    ('gauss', GaussianProcessRegressor()) #Specify via GAUSSIANPROCESSREGRESSOR
]

# define the parameter space for the grid search
GS_PARAMS = {
    "svr__epsilon": [0, 0.01, 0.1, 0.3],
    "svr__C": [1],
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


#class for out custom stacked regressor
#Seperate class since I am not sure about the implementation of GridSearch on the Stacked regressor, this
# we can write a GridSearch that optimizes each estimator on its own since the parameter space for the whole
# StackedRegressor might be to large
class StackedRegressor():

    #initialized the regrassor given the ESTIMATORS parameters
    def __init__(self):
        self.regressor = StackingRegressor(estimators= ESTIMATORS , final_estimator=RidgeCV())

    #performs grid search on the parameter space
    def gridsearch(self,x_data,y_data):
        grid = GridSearchCV(estimator=self.regressor, param_grid=GS_PARAMS, cv=5)
        grid.fit(x_data,y_data)
        self.best_params = grid.get_params()

    #fit the regressor by manually handing it the best parameters
    def fit(self,x_data,y_data,params):
        self.regressor.set_params(params)
        self.regressor.fit(x_data,y_data)

    # fit the regressor by atomatically choosing the best parameters
    # Gridsearch has to be run before this function can be used
    def fit(self,x_data,y_data):
        try:
            self.regressor.set_params(self.best_params)
            self.regressor.fit(x_data,y_data)
        except:
            print("Gridsearch has not yet been performed!")

    def make_prediction(self,x_data):
        return self.regressor.predict(x_data)

    def get_params(self):
        return self.regressor.get_params()
