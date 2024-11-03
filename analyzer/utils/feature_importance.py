#utils/feature_importance.py
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class FeatureImportanceAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def analyze_feature_importance(self, model, feature_cols: List[str]) -> None:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("Modelo no soporta analisis de feature importance.")
            return

        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        self.plot_feature_importance(feature_importance)
        self.save_feature_importance(feature_importance)

    def plot_feature_importance(self, feature_importance: List[Tuple[str, float]], top_n: int = 20) -> None:
        top_features = feature_importance[:top_n]
        names, values = zip(*top_features)

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, names)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} features más importantes')
        
        for i, v in enumerate(values):
            plt.text(v, i, f'{v:.4f}', va='center')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()

    def save_feature_importance(self, feature_importance: List[Tuple[str, float]]) -> None:
        df = pd.DataFrame(feature_importance, columns=['Feature', 'Importancia'])
        df.to_csv(self.output_dir / 'feature_importance.csv', index=False)

    def analyze_feature_correlations(self, df: pd.DataFrame, feature_cols: List[str], 
                                   target_col: str = 'cost_of_living') -> None:
        correlations = []
        for col in feature_cols:
            corr = df[col].corr(df[target_col])
            correlations.append((col, corr))
            
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlación'])
        corr_df.to_csv(self.output_dir / 'feature_correlations.csv', index=False)
        
