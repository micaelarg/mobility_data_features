# analyzer/analyzer.py
from pathlib import Path
import pandas as pd
import os
from typing import List

from .core import HexProcessor
from .features import (
    TemporalFeatureCalculator,
    DeviceFeatureCalculator,
    DistanceCalculator
)
from .data_processor import ChunkProcessor

class AltScoreAnalyzer:
    def __init__(self, cities_data, commercial_areas, universities, tourist_spots, 
                 geographic_features, output_dir=Path('./results')):
        self.hex_processor = HexProcessor()
        self.chunk_processor = ChunkProcessor(self.hex_processor, output_dir)
        self.temporal_calculator = TemporalFeatureCalculator()
        self.device_calculator = DeviceFeatureCalculator()
        self.distance_calculator = DistanceCalculator(
            cities_data, universities, tourist_spots, geographic_features
        )
        self.output_dir = Path(output_dir)

    def _process_hex_groups(self, chunk: pd.DataFrame) -> List[dict]:
        hex_features = []
        
        print("\nInput columns chunk:", chunk.columns.tolist())
        
        # groupby object
        grouped = chunk.groupby('hex_id')
        
        for hex_id, group in grouped:
            try:
                print(f"\nProcesando grupo de hex_id {hex_id}")
                print(f"Columnas del grupo: {group.columns.tolist()}")
                
                if isinstance(group, pd.Series):
                    group = group.to_frame()
                
                features = self._calculate_hex_features(hex_id, group)
                
                if features:
                    print(f"Calculando feature keys: {features.keys()}")
                    hex_features.append(features)
                else:
                    print(f"hex_is {hex_id} no tiene features calculadas")
    
            except Exception as e:
                print(f"Error procesando hex_id {hex_id}: {str(e)}")
                continue
        
        print(f"\n{len(hex_features)} hex grupos procesados")
        return hex_features
    

    def _prepare_group_data(self, group: pd.DataFrame) -> pd.DataFrame:
        try:
            group = group.copy()
            
            if 'timestamp' in group.columns:
                group['timestamp'] = pd.to_datetime(group['timestamp'])
                if 'hour' not in group.columns:
                    group['hour'] = group['timestamp'].dt.hour
                if 'day_of_week' not in group.columns:
                    group['day_of_week'] = group['timestamp'].dt.dayofweek + 1
                    
            # Debug: check required columns
            required_columns = ['timestamp', 'hour', 'day_of_week']
            missing_columns = [col for col in required_columns if col not in group.columns]
            if missing_columns:
                print(f"Missing cvolumnas necesarias: {missing_columns}")
                return None
                
            return group
            
        except Exception as e:
            print(f"Error preparando la datas del hex_id grupo: {str(e)}")
            return None
    
    def _calculate_hex_features(self, hex_id: str, group: pd.DataFrame) -> dict:
        if len(group) == 0:
            return None
    
        print(f"\nCalculando features para el hex_id {hex_id}")
        print(f"Columnas available: {group.columns.tolist()}")
        
        try:
            temporal_columns = [
                'weekend_ratio', 'weekday_ratio', 'night_ratio',
                'rush_hour_ratio', 'business_hours_ratio',
                'morning_ratio', 'afternoon_ratio', 'evening_ratio',
                'total_records'
            ]
            
            device_columns = [
                'unique_devices', 'records_per_device', 'device_diversity',
                'frequent_devices', 'very_frequent_devices'
            ]
            
            has_temporal = all(col in group.columns for col in temporal_columns)
            has_device = all(col in group.columns for col in device_columns)
            
            if has_temporal and has_device:
                features = {
                    'hex_id': hex_id,
                    'lat': float(group['lat'].iloc[0]),
                    'lon': float(group['lon'].iloc[0]),
                    'day_of_week': int(group['day_of_week'].iloc[0]),
                    'weekend_ratio': float(group['weekend_ratio'].iloc[0]),
                    'weekday_ratio': float(group['weekday_ratio'].iloc[0]),
                    'night_ratio': float(group['night_ratio'].iloc[0]),
                    'rush_hour_ratio': float(group['rush_hour_ratio'].iloc[0]),
                    'business_hours_ratio': float(group['business_hours_ratio'].iloc[0]),
                    'morning_ratio': float(group['morning_ratio'].iloc[0]),
                    'afternoon_ratio': float(group['afternoon_ratio'].iloc[0]),
                    'evening_ratio': float(group['evening_ratio'].iloc[0]),
                    'total_records': int(group['total_records'].iloc[0]),
                    'unique_devices': int(group['unique_devices'].iloc[0]),
                    'records_per_device': float(group['records_per_device'].iloc[0]),
                    'device_diversity': float(group['device_diversity'].iloc[0]),
                    'frequent_devices': int(group['frequent_devices'].iloc[0]),
                    'very_frequent_devices': int(group['very_frequent_devices'].iloc[0])
                }
            else:
                temporal_features = self.temporal_calculator.calculate_temporal_features(group)
                if temporal_features is None:
                    return None
                    
                device_features = self.device_calculator.calculate_device_features(group)
                
                features = {
                    'hex_id': hex_id,
                    'lat': group['lat'].mean(),
                    'lon': group['lon'].mean(),
                    **temporal_features,
                    **device_features
                }
            
            derived_features = self._calculate_derived_features(features)
            features.update(derived_features)
            
            print(f"Feature keys calculadas: {features.keys()}")
            return features
            
        except Exception as e:
            print(f"Error calculando features para hex_id {hex_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def _calculate_derived_features(self, features: dict) -> dict:
        total_records = features.get('total_records', 0)
        if total_records == 0:
            return {}
    
        business_hours_ratio = features.get('business_hours_ratio', 0)
        weekend_ratio = features.get('weekend_ratio', 0)
        night_ratio = features.get('night_ratio', 0)
        unique_devices = features.get('unique_devices', 0)
            
        return {
            'commercial_activity_score': (
                business_hours_ratio * 
                unique_devices / total_records
            ),
            'residential_activity_score': (
                (night_ratio + weekend_ratio) / 2 * 
                unique_devices / total_records
            )
        }
    
    def _aggregate_chunks(self, chunk_files: List[Path]) -> pd.DataFrame:
        all_features = []
        
        for chunk_file in chunk_files:
            try:
                chunk = pd.read_parquet(chunk_file)
                print(f"\nProcesando chunk: {chunk_file} de shape: {chunk.shape}")
                hex_features = self._process_hex_groups(chunk)
                
                # check features before adding
                if hex_features:
                    for features in hex_features:
                        if 'day_of_week' not in features:
                            print(f"day_of_week missing de features para el hex_id {features.get('hex_id')}")
                            continue
                        all_features.append(features)
                
                print(f"Features extraídas del chunk {chunk_file}: {len(hex_features)}")
    
            except Exception as e:
                print(f"Error procesando chunk {chunk_file}: {str(e)}")
                continue
        
        if not all_features:
            print("No se extrajeron features de ningún chunk.")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(all_features)
        
        if 'day_of_week' in features_df.columns:
            features_df['day_of_week'] = features_df['day_of_week'].astype(int)
            print("\nday_of_week distribución:")
            print(features_df['day_of_week'].value_counts())
        else:
            print("\nday_of_week está missing del final DataFrame!")
            print("Columns:", features_df.columns.tolist())
        
        return features_df
        

    def _calculate_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("\nCalculando features de distancia")
            print(f"Input DataFrame shape: {df.shape}")
            print(f"Input columns: {df.columns.tolist()}")
            
            batch_size = 100000
            final_features = []
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}")
                print(f"Batch size: {len(batch)}")
                
                try:
                    processed_batch = self.distance_calculator.calculate_all_distances(batch)
                    
                    new_columns = [col for col in processed_batch.columns if col not in df.columns]
                    print(f"Distance features agregadas: {new_columns}")
                    
                    final_features.append(processed_batch)
                    
                except Exception as e:
                    print(f"Error procesando batch {i//batch_size + 1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not final_features:
                print("No se calculó ninguna feature de distancia!")
                return df
                
            result_df = pd.concat(final_features, ignore_index=True)
            
            print("\nFinal results:")
            print(f"Shape: {result_df.shape}")
            print(f"Todas las columnas: {result_df.columns.tolist()}")
            print("\nEstadísticas de las features de distancia:")
            distance_cols = [col for col in result_df.columns if col not in df.columns]
            for col in distance_cols:
                print(f"\n{col}:")
                print(result_df[col].describe())
            
            return result_df
            
        except Exception as e:
            print(f"Error en _calculate_distance_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return df
    
    def analyze_data(self, input_path: str) -> pd.DataFrame:
        chunk_files = [] 
        try:
            chunk_files = self.chunk_processor.process_parquet_to_chunks(input_path)  
            features_df = self._aggregate_chunks(chunk_files)
            
            if features_df.empty:
                print("ERROR: No se extrajeron features!")
                return pd.DataFrame()
                
            print("\nFeatures generadas:")
            print(f"Shape: {features_df.shape}")
            print(f"Columns: {features_df.columns.tolist()}")
            
            print("\nCalculando features de distancia")
            final_features_df = self._calculate_distance_features(features_df)
            
            expected_distance_features = [
                'nearest_city_dist', 'weighted_city_importance', 'weighted_elevation',
                'nearest_uni_dist', 'weighted_uni_importance',
                'nearest_tourist_spot_dist', 'weighted_tourist_importance',
                'coast_distance', 'nearest_mountain_dist', 'weighted_mountain_influence'
            ]
            
            present_features = [col for col in expected_distance_features if col in final_features_df.columns]
            print(f"\nFeatures de distancia: {present_features}")
            
            return final_features_df
            
        finally:
            self._cleanup_temp_files(chunk_files)

    def _cleanup_temp_files(self, chunk_files: List[Path]) -> None:
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
            except Exception as e:
                print(f"Error eliminando chunk {chunk_file}: {str(e)}")

        temp_dir = self.output_dir / "temp_chunks"
        try:
            if temp_dir.exists():
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error eliminando directorio {temp_dir}: {str(e)}")
