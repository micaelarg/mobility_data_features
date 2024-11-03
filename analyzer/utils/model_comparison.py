#utils/mode_comparison.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm.auto import tqdm
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import GridSearchCV, train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor



def run_grid_search(estimator, param_grid, X_train, y_train, model_name=""):
    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=4, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


class ModelComparer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.models = {}
        self.metrics = {}
        
    def split_train_val(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(data, test_size=test_size, random_state=6)
        
    def add_non_linear_features(self, data: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        transformations = {}
        for feature in selected_features:
            transformations[f"{feature}_squared"] = data[feature] ** 2
            transformations[f"{feature}_cubed"] = data[feature] ** 3 
            transformations[f"{feature}_sqrt"] = np.sqrt(np.abs(data[feature]))
            transformations[f"{feature}_log"] = np.log1p(np.abs(data[feature]))
    
        transformed_features = pd.DataFrame(transformations, index=data.index)
        data = pd.concat([data, transformed_features], axis=1)
        
        return data


    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, feature_cols: List[str]) -> Dict:
        print("\nPreparando la data...")
        
        if 'cost_of_living' not in train_data.columns or 'cost_of_living' not in val_data.columns:
            raise ValueError("The 'cost_of_living' column is missing from train or validation data.")
        
        train_data['cost_of_living_log'] = np.log1p(train_data['cost_of_living'])
        val_data['cost_of_living_log'] = np.log1p(val_data['cost_of_living'])
    
        y_train = train_data['cost_of_living_log'].copy()
        y_val = val_data['cost_of_living_log'].copy()
    
        feature_cols = [col for col in feature_cols if col not in ['hex_id', 'cost_of_living', 'total_records', 'cost_of_living_log']]
        
        selected_features = [col for col in feature_cols if train_data[col].dtype in ['float64', 'int64']]
        
        train_data = self.add_non_linear_features(train_data[feature_cols], selected_features)
        val_data = self.add_non_linear_features(val_data[feature_cols], selected_features)
        
        feature_cols.extend([f"{feature}_squared" for feature in selected_features])
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data)
        X_val = scaler.transform(val_data)
        
        self.feature_transformer = scaler
        self.feature_cols = feature_cols
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'scaler': scaler,
            'feature_names': feature_cols
        }
    
    def prepare_test_data(self, test_data: pd.DataFrame) -> np.ndarray:
        test_data = self.add_non_linear_features(test_data[self.feature_cols], self.feature_cols)
        test_data_scaled = self.feature_transformer.transform(test_data)
        
        return test_data_scaled

    
    def evaluate_model(self, model, X_val, y_val_log) -> Dict[str, float]:
        predictions_log = model.predict(X_val)
        predictions = np.expm1(predictions_log)  
        y_val = np.expm1(y_val_log)  
    
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
    
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
        
    def compare_models(self, data: pd.DataFrame, feature_cols: List[str] = None):
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col not in ['hex_id', 'cost_of_living', 'total_records']]
        
        train_data, val_data = self.split_train_val(data)
        prepared_data = self.prepare_data(train_data, val_data, feature_cols)
        
        models_to_train = {
            'gbm': self.train_gbm,
            'xgboost': self.train_xgboost,
            'lightgbm': self.train_lightgbm,
            'random_forest': self.train_random_forest,
        }
        
        for model_name, train_func in models_to_train.items():
            try:
                self.metrics[model_name] = train_func(
                    prepared_data['X_train'],
                    prepared_data['y_train'],
                    prepared_data['X_val'],
                    prepared_data['y_val'],
                )
                self.metrics[model_name] = self.evaluate_model(
                    self.models[model_name],  
                    prepared_data['X_val'],
                    prepared_data['y_val']
                )
            except Exception as e:
                print(f"Error entrenando {model_name}: {str(e)}")
                self.metrics[model_name] = {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': float('-inf')
                }
        
        print("\nCreando plots comparativos")
        self.plot_model_comparison()
    
        best_model = min(self.metrics.items(), key=lambda x: x[1]['rmse'])
        return best_model[0], self.models[best_model[0]], self.metrics
    

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def plot_model_comparison(self):
        metrics_df = pd.DataFrame(self.metrics).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.metrics))
        width = 0.25
        
        ax.bar(x - width, [m['rmse'] for m in self.metrics.values()], width, label='RMSE')
        ax.bar(x, [m['mae'] for m in self.metrics.values()], width, label='MAE')
        ax.bar(x + width, [m['r2'] for m in self.metrics.values()], width, label='R²')
        
        ax.set_ylabel('Score')
        ax.set_title('Comparación de Modelos')
        ax.set_xticks(x)
        ax.set_xticklabels(self.metrics.keys())
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png')
        plt.close()
        
        print("\nMetrics - Comparación:")
        print(metrics_df.round(4))

    def train_gbm(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        print("\nEntrenando GBM")
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [5, 10, 20],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        }
        
        model = GradientBoostingRegressor(
        random_state=6,
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=1e-4
        )
        best_model = run_grid_search(model, param_grid, X_train, y_train, model_name="GBM")
        predictions = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, predictions)
        self.models['gbm'] = best_model
    
        print(f"\nBest parámetros GBM: {best_model.get_params()}")
        return metrics

    def train_xgboost(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        print("\nEntrenando XGBoost")
        
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=6,
            early_stopping_rounds=10,
            eval_metric='rmse',
            verbosity=0,
            n_jobs=4
        )
    
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=4, verbose=1)
        grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, predictions)
        self.models['xgboost'] = best_model
        
        return metrics

    def train_random_forest(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        print("\nEntrenando Random Forest")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        model = RandomForestRegressor(random_state=6)
        best_model = run_grid_search(model, param_grid, X_train, y_train, model_name="Random Forest")
        predictions = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, predictions)
        self.models['random_forest'] = best_model
        
        return metrics
     

    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        print("\nEntrenando LightGBM")
        
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        model = LGBMRegressor(random_state=6)
        best_model = run_grid_search(model, param_grid, X_train, y_train, model_name="LightGBM")
        predictions = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, predictions)
        self.models['lightgbm'] = best_model
        
        return metrics


