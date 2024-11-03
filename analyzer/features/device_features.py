import pandas as pd
import numpy as np

class DeviceFeatureCalculator:
    @staticmethod
    def calculate_device_features(group: pd.DataFrame) -> dict:
        total_records = len(group)
        unique_devices = group['device_id'].nunique()
        device_counts = group['device_id'].value_counts()
        
        return {
            'total_records': total_records,
            'unique_devices': unique_devices,
            'records_per_device': total_records / unique_devices,
            'device_diversity': unique_devices / total_records,
            'frequent_devices': (device_counts > np.percentile(device_counts, 75)).sum(),
            'very_frequent_devices': (device_counts > np.percentile(device_counts, 90)).sum(),
        }
