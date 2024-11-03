# Mobility Data Analysis

## Overview
This project analyzes mobility data using hexagonal grid mapping (H3) to calculate various features and predict cost of living indices in cities from Ecuador. The analysis includes temporal patterns, device usage, and geographical features.

## Project Structure
```
.
├── analyzer/               
│   ├── core/            
│   │   └── hex_processor.py
│   ├── data_processor/   
│   │   └── chunk_processor.py
│   ├── features/          
│   │   ├── device_features.py
│   │   ├── distance_calculator.py
│   │   └── temporal_features.py
│   └── utils/           
│       ├── feature_importance.py
│       └── model_comparison.py
├── data/                  
│   ├── external_data/     
│   ├── mobility_data.parquet
│   ├── test.csv
│   └── train.csv
├── results/            
└── main.py              
```

## Features
- Hexagonal grid analysis using H3 indexing
- Temporal pattern analysis (weekday/weekend, hourly patterns)
- Device usage patterns
- Geographic feature distance calculations
- Machine learning models for cost prediction
- Feature importance analysis

## Requirements
Check requirements.txt file.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/micaelarg/mobility_data_features
cd altscore
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your data files in the `data/` directory:
   - mobility_data.parquet
   - train.csv
   - test.csv

Example files can be found in kaggle competitions download -c alt-score-data-science-competition

2. Run the analysis:
```bash
python main.py
```

3. Check results in the `results/` directory:
   - analyzed_features.parquet: Extracted features
   - model_comparison.png: Model performance visualization
   - predictions.csv: Cost predictions
   - summary_stats.csv: Analysis summary

## Feature Extraction
The project extracts several types of features:

### Temporal Features
- Weekend/weekday ratios
- Time-of-day patterns
- Rush hour activity
- Business hours patterns

### Device Features
- Unique device counts
- Records per device
- Device diversity metrics
- Frequency patterns

### Geographic Features
- Distance to nearest cities
- Proximity to universities
- Tourist spot influence
- Geographic feature distances

## Models
The project compares two models:
- Gradient Boosting Machine (GBM)
- XGBoost
- Linear Regression
- LightGBM

Models are evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score

## Output
The analysis produces:
1. Feature importance rankings
2. Model comparison metrics
3. Cost of living predictions
4. Summary statistics (output can be improved)

## Debugging
If you encounter path-related issues:
```python
print(f"\nChecking paths:")
print(f"Base directory: {base_dir}")
print(f"Data directory: {data_dir}")
print(f"Input path exists: {os.path.exists(input_path)}")
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[MIT License
Copyright (c) 2024 Micaela Kulesz
]

## Contact
You can contact me on LinkedIn --> www.linkedin.com/in/micaelamaria

