import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor # Added LightGBM
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Note: The array has two targets at the end: price and price_per_sqft.
            # We slice [:, :-2] to get all features, and [:, -2] to grab just the primary 'price' target.
            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],
                train_array[:, -2], 
                test_array[:, :-2],
                test_array[:, -2]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False,allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LightGBM Regressor": LGBMRegressor() # Added LightGBM
            }

            # Expanded Hyperparameter Grids
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 50],
                    'min_samples_split': [2, 5, 10],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'n_estimators': [32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 9]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'n_estimators': [32, 64, 128, 256],
                    'max_depth': [3, 5, 7, 9],
                    'min_child_weight': [1, 3, 5]
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100, 200]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [32, 64, 128, 256]
                },
                "LightGBM Regressor": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [32, 64, 128, 256],
                    'max_depth': [-1, 5, 10, 20],
                    'num_leaves': [31, 50, 100]
                }
            }

            # Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found (Score < 0.6)")
                
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Save the winning model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Calculate final R2 score on the test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            n = X_test.shape[0]  # Number of samples (rows)
            p = X_test.shape[1]  # Number of features (columns)
            adjusted_r2 = 1 - (1 - r2_square) * (n - 1) / (n - p - 1)

            logging.info(f"Model Training Complete.")
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Test R2 Score: {r2_square}")
            logging.info(f"Test Adjusted R2 Score: {adjusted_r2}")

            # Return both metrics so you can log or display them later
            return r2_square, adjusted_r2
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)