# analyzer/data_processor/chunk_processor.py
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import List
from typing import Dict, Optional


class ChunkProcessor:
    def __init__(self, hex_processor, output_dir: Path):
        self.hex_processor = hex_processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_parquet_to_chunks(self, parquet_path: str, chunk_size: int = 50000) -> List[Path]:
        temp_dir = self.output_dir / "temp_chunks"
        temp_dir.mkdir(exist_ok=True)
        
        print("\nLeyendo mobility_data")
        parquet_file = pq.ParquetFile(parquet_path)
        num_row_groups = parquet_file.num_row_groups
        
        print(f"{num_row_groups} grupos encontrados")
        chunk_files = []
        
        for group_idx in range(num_row_groups):
            try:
                print(f"\nProcesando grupo {group_idx + 1}/{num_row_groups}")
                
                chunk = parquet_file.read_row_group(
                    group_idx, 
                    columns=['lat', 'lon', 'timestamp', 'device_id']
                ).to_pandas()
                
                processed_chunk = self._process_chunk(chunk)
                if processed_chunk is None or len(processed_chunk) == 0:
                    continue
                
                output_file = temp_dir / f"group_{group_idx}.parquet"
                processed_chunk.to_parquet(output_file)
                chunk_files.append(output_file)
                
            except Exception as e:
                print(f"Error procesando grupo {group_idx}: {str(e)}")
                continue
                
        return chunk_files

    def process_coordinates_batch(self, df_batch: pd.DataFrame) -> List[str]:
        return df_batch.apply(lambda row: self.hex_processor.calculate_hex_id(row['lat'], row['lon']), axis=1).tolist()

    def _process_chunk(self, chunk_data: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk_data.dropna(subset=['lat', 'lon'])
        if len(chunk) == 0:
            return None
    
        if 'device_id' not in chunk.columns:
            print("Columna 'device_id' no está en el chunk - salteo el proceso.")
            return None

        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
    
        if chunk['timestamp'].isnull().any():
            print("Hay timestamps que no se pudieron convertir a datetime")
    
        # Debug: check si timestamp anduvo
        chunk['hour'] = chunk['timestamp'].dt.hour
        chunk['day_of_week'] = chunk['timestamp'].dt.dayofweek + 1  # Adjust for 1-7
    
        # Debug: check el df arrojado
        print("columnas de los chunks procesados")
        print(chunk.columns)
        print(chunk[['timestamp', 'hour', 'day_of_week']].head())
    
        batch_size = 10000
        hex_ids = []
        for i in range(0, len(chunk), batch_size):
            batch = chunk.iloc[i:i + batch_size]
            hex_ids.extend(self.process_coordinates_batch(batch))
    
        chunk['hex_id'] = hex_ids
        return chunk.dropna(subset=['hex_id'])

