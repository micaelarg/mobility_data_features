# analyzer/core/hex_processor.py
import pandas as pd
import h3
from pathlib import Path

class HexProcessor:
    def __init__(self, resolution=8):
        self.resolution = resolution

    def calculate_hex_id(self, lat: float, lon: float) -> str:
        try:
            if pd.isna(lat) or pd.isna(lon):
                return None
            return h3.latlng_to_cell(lat, lon, self.resolution)
        except Exception as e:
            print(f"Error calculando hex_id for lat={lat}, lon={lon}: {str(e)}")
            return None

    def process_coordinates_batch(self, df_batch: pd.DataFrame) -> pd.Series:
        return df_batch.apply(lambda row: self.calculate_hex_id(row['lat'], row['lon']), axis=1)
