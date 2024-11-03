import numpy as np
import pandas as pd

class DistanceCalculator:
    def __init__(self, cities_data, universities, tourist_spots, geographic_features):
        self.cities_data = cities_data
        self.universities = universities
        self.tourist_spots = tourist_spots
        self.geographic_features = geographic_features

    def calculate_all_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nCalculando features de distancia")
        result = df.copy()
        points = np.column_stack((result['lat'], result['lon']))
        
        city_features = self._calculate_city_features(points)
        uni_features = self._calculate_university_features(points)
        tourist_features = self._calculate_tourist_features(points)
        geo_features = self._calculate_geographic_features(points)
        
        for feature_dict in [city_features, uni_features, tourist_features, geo_features]:
            for key, values in feature_dict.items():
                result[key] = values
                print(f"Features agregadas: {key}")
        
        new_columns = [col for col in result.columns if col not in df.columns]
        print(f"\nAgregadas {len(new_columns)} nuevas features de distancia: {new_columns}")
        
        return result

    def _calculate_city_features(self, points):
        print("Calculando features de la ciudad.")
        if not self.cities_data:
            print("No hay data available de ciudad")
            return {}
            
        city_coords = np.array([city['coords'] for city in self.cities_data.values()])
        city_weights = np.array([
            float(city['population']) * float(city['gdp_per_capita']) 
            for city in self.cities_data.values()
        ])
        city_elevations = np.array([float(city['elevation']) for city in self.cities_data.values()])
        city_distances = np.sqrt(((points[:, np.newaxis, :] - city_coords) ** 2).sum(axis=2))
        
        features = {
            'nearest_city_dist': city_distances.min(axis=1),
            'weighted_city_importance': (city_weights / (city_distances + 0.001)).sum(axis=1),
            'weighted_elevation': (city_elevations / (city_distances + 0.001)).sum(axis=1)
        }
        print(f"Features de ciudad: {list(features.keys())}")
        return features

    def _calculate_university_features(self, points):
        print("Calculando features de universidad")
        if not self.universities:
            print("No hay data available de universidades")
            return {}
            
        uni_coords = np.array([uni['coords'] for uni in self.universities])
        uni_weights = np.array([1.0 / float(uni['ranking']) for uni in self.universities])
        uni_distances = np.sqrt(((points[:, np.newaxis, :] - uni_coords) ** 2).sum(axis=2))
        
        features = {
            'nearest_uni_dist': uni_distances.min(axis=1),
            'weighted_uni_importance': (uni_weights / (uni_distances + 0.001)).sum(axis=1)
        }
        print(f"Features de universidades: {list(features.keys())}")
        return features

    def _calculate_tourist_features(self, points):
        print("Calculando features de turismo")
        if not self.tourist_spots:
            print("No hay data available de turismo")
            return {}
            
        tourist_coords = np.array([spot['coords'] for spot in self.tourist_spots])
        tourist_weights = np.array([
            1.0 if spot['importance'] == 'high' else 0.5 
            for spot in self.tourist_spots
        ])
        tourist_distances = np.sqrt(((points[:, np.newaxis, :] - tourist_coords) ** 2).sum(axis=2))
        
        features = {
            'nearest_tourist_spot_dist': tourist_distances.min(axis=1),
            'weighted_tourist_importance': (tourist_weights / (tourist_distances + 0.001)).sum(axis=1)
        }
        print(f"Features de turismo calculadas: {list(features.keys())}")
        return features

    def _calculate_geographic_features(self, points): #rearmé con ifs porque es más claro
        print("Calculando features geográficas")
        features = {}
        
        if 'coast_line' in self.geographic_features and self.geographic_features['coast_line']:
            coast_coords = np.array(self.geographic_features['coast_line'])
            coast_distances = np.sqrt(((points[:, np.newaxis, :] - coast_coords) ** 2).sum(axis=2))
            features['coast_distance'] = coast_distances.min(axis=1)
        
        if 'mountains' in self.geographic_features and self.geographic_features['mountains']:
            mountain_coords = np.array([m['coords'] for m in self.geographic_features['mountains']])
            mountain_elevations = np.array([float(m['elevation']) for m in self.geographic_features['mountains']])
            mountain_distances = np.sqrt(((points[:, np.newaxis, :] - mountain_coords) ** 2).sum(axis=2))
            
            features.update({
                'nearest_mountain_dist': mountain_distances.min(axis=1),
                'weighted_mountain_influence': (mountain_elevations / (mountain_distances + 0.001)).sum(axis=1)
            })
        
        print(f"Features geográficas: {list(features.keys())}")
        return features
