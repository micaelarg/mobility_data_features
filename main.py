# main.py
from pathlib import Path
import os
import pandas as pd
from analyzer import AltScoreAnalyzer, FeatureImportanceAnalyzer
from data.external_data.location_data import (
    cities_data,
    commercial_areas,
    universities,
    tourist_spots,
    geographic_features
)
from analyzer.utils.model_comparison import ModelComparer


def main():

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'results')
    
    input_path = os.path.join(data_dir, 'mobility_data.parquet')
    train_data_path = os.path.join(data_dir, 'train.csv')
    test_data_path = os.path.join(data_dir, 'test.csv')
    
    os.makedirs(output_dir, exist_ok=True)

    print("Inicializando")
    analyzer = AltScoreAnalyzer(
        cities_data=cities_data,
        commercial_areas=commercial_areas,
        universities=universities,
        tourist_spots=tourist_spots,
        geographic_features=geographic_features,
        output_dir=output_dir
    )

    print("Comenzando análisis")
    try:
        result_df = analyzer.analyze_data(input_path)
        
        output_path = os.path.join(output_dir, 'analyzed_features.parquet')
        result_df.to_parquet(output_path)
        print(f"Análisis completado. Resultados en {output_path}")

        summary_stats = {
            'total_hexagons': len(result_df),
            'unique_devices': result_df['unique_devices'].sum(),
            'avg_commercial_score': result_df['commercial_activity_score'].mean(),
            'avg_residential_score': result_df['residential_activity_score'].mean(),
            'total_cities_analyzed': len(cities_data),
            'total_commercial_areas': len(commercial_areas),
            'total_universities': len(universities),
            'total_tourist_spots': len(tourist_spots),
            'total_mountains': len(geographic_features['mountains']),
            'coastline_points': len(geographic_features['coast_line'])
        }
        
        summary_path = os.path.join(output_dir, 'summary_stats.csv')
        pd.DataFrame([summary_stats]).to_csv(summary_path)
        print("Resument stats guardado")
        
        # duplicate_hex_ids = result_df[result_df.duplicated(subset=['hex_id'], keep=False)]
        # num_duplicates = duplicate_hex_ids['hex_id'].nunique()
        # duplicate_hex_sample = duplicate_hex_ids.head()
        result_df = result_df.groupby('hex_id').mean().reset_index()

        
        print("\nSubiendo y mergeando train y test")
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        
        result_df_hex = set(result_df['hex_id'].unique())
        train_df = train_df[train_df['hex_id'].isin(result_df_hex)]
        test_df = test_df[test_df['hex_id'].isin(result_df_hex)]
       
        train_with_features = train_df.merge(result_df, on='hex_id', how='left')
        test_with_features = test_df.merge(result_df, on='hex_id', how='left')

        print("\nModelando")
        model_comparer = ModelComparer(output_dir=output_dir)

        feature_cols = [col for col in train_with_features.columns 
                       if col not in ['hex_id', 'cost_of_living']]
        
        best_model_name, best_model, metrics = model_comparer.compare_models(
            data=train_with_features, 
            feature_cols=feature_cols
        )
        print(f"\nBest model: {best_model_name}")

        print("\nFeature importance")
        feature_analyzer = FeatureImportanceAnalyzer(output_dir=output_dir)
        feature_analyzer.analyze_feature_importance(best_model, feature_cols)

        print("\nPrediciendo")
        test_features = test_with_features[feature_cols]
        
        if 'total_records' in test_features.columns:
            test_features = test_features.drop(columns=['total_records'])

        test_features = model_comparer.prepare_test_data(test_with_features)
        predictions = best_model.predict(test_features)

        submission_df = pd.DataFrame({
            'hex_id': test_with_features['hex_id'],
            'cost_of_living': predictions
        })
        
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir.mkdir(parents=True, exist_ok=True) 
        submission_path = output_dir / 'predictions.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Predicciones en {submission_path}")

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()