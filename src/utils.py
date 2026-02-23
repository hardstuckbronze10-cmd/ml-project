import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from skopt import BayesSearchCV # Imported BayesSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para = param[model_name]

            # Skip the search if the parameter grid is empty (e.g., Linear Regression)
            if not para:
                model.fit(X_train, y_train)
            else:
                # Replaced GridSearchCV with BayesSearchCV
                bs = BayesSearchCV(
                    estimator=model, 
                    search_spaces=para, 
                    n_iter=15, # Controls how many parameter combinations it will try
                    cv=3,
                    n_jobs=-1, # Uses all available CPU cores
                    random_state=42
                )
                bs.fit(X_train, y_train)

                # Set the best parameters found by the Bayesian search
                model.set_params(**bs.best_params_)
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)