import pandas as pd
import numpy as np

class TemporalFeatureCalculator:
    @staticmethod
    def calculate_temporal_features(group: pd.DataFrame) -> dict:
        print("\nCalculando features temporales")
        print(f"Input columns: {group.columns.tolist()}")
        
        total_records = len(group)
        if total_records == 0:
            return None
            
        try:
            if 'timestamp' in group.columns:
                if not pd.api.types.is_datetime64_any_dtype(group['timestamp']):
                    group['timestamp'] = pd.to_datetime(group['timestamp'])
                
                group['hour'] = group['timestamp'].dt.hour
            elif 'hour' not in group.columns:
                print("Warning: Neither timestamp nor hour found in data")
                return None
                
            if 'day_of_week' not in group.columns and 'timestamp' in group.columns:
                group['day_of_week'] = group['timestamp'].dt.dayofweek + 1  # 1-7 for Monday-Sunday
            
            # check columnas necesarias
            print(f"Processed columns: {group.columns.tolist()}")
            if 'hour' not in group.columns or 'day_of_week' not in group.columns:
                print("Required columns missing after processing")
                return None
                
            day_of_week = int(group['day_of_week'].mode()[0])
            weekend_mask = group['day_of_week'].isin([6, 7])
            night_mask = (group['hour'] >= 22) | (group['hour'] <= 6)
            rush_hour_mask = group['hour'].isin([7, 8, 9, 17, 18, 19])
            business_hours_mask = (group['hour'] >= 9) & (group['hour'] <= 18)
            morning_mask = (group['hour'] >= 6) & (group['hour'] <= 11)
            afternoon_mask = (group['hour'] >= 12) & (group['hour'] <= 17)
            evening_mask = (group['hour'] >= 18) & (group['hour'] <= 21)   
            weekend_count = weekend_mask.sum()
            weekday_count = total_records - weekend_count
            
            features = {
                'day_of_week': day_of_week,
                'weekend_ratio': weekend_count / total_records,
                'weekday_ratio': weekday_count / total_records,
                'night_ratio': night_mask.sum() / total_records,
                'rush_hour_ratio': rush_hour_mask.sum() / total_records,
                'business_hours_ratio': business_hours_mask.sum() / total_records,
                'morning_ratio': morning_mask.sum() / total_records,
                'afternoon_ratio': afternoon_mask.sum() / total_records,
                'evening_ratio': evening_mask.sum() / total_records,
                'total_records': total_records
            }
            
            print(f"Features temporales calculadas: {list(features.keys())}")
            return features
            
        except Exception as e:
            print(f"Error calculando features temporales: {str(e)}")
            import traceback
            traceback.print_exc()
            return None