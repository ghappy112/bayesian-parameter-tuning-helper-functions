import xgboost as xgb

from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score

import numpy as np



def xgb_evaluate(max_depth, gamma, colsample_bytree, subsample, n_estimators):

    params = {

        'booster': 'gbtree',

        'max_depth': int(max_depth),

        'gamma': gamma,

        'colsample_bytree': colsample_bytree,

        'subsample': subsample,

        'n_estimators': int(n_estimators),

        'objective': 'binary:logistic',

        'eval_metric': 'auc',

        'silent': True

    }

    # Using cross validation for evaluation

    cv_result = xgb.cv(

        params,

        X,

        y,

        nfold=5,

        metrics='auc',

        num_boost_round=200,

        early_stopping_rounds=10

    )

    return cv_result['test-auc-mean'].iloc[-1]



# Define the bounds for each hyperparameter

bounds = {

    'max_depth': (3, 10),

    'gamma': (0, 1),

    'colsample_bytree': (0.1, 1),

    'subsample': (0.1, 1),

    'n_estimators': (100, 1000)

}



# Pass the evaluation function to BayesianOptimization

optimizer = BayesianOptimization(

    f=xgb_evaluate,

    pbounds=bounds,

    random_state=7,

    verbose=2

)



# Run the optimization

optimizer.maximize(init_points=10, n_iter=50)



# Get the best hyperparameters found by the optimization

best_params = optimizer.max['params']



# Train the XGBoost model with the best hyperparameters

model = xgb.XGBClassifier(

    booster='gbtree',

    max_depth=int(best_params['max_depth']),

    gamma=best_params['gamma'],

    colsample_bytree=best_params['colsample_bytree'],

    subsample=best_params['subsample'],

    n_estimators=int(best_params['n_estimators']),

    objective='binary:logistic'

)

model.fit(X, y)
