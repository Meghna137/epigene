import numpy as np
import pandas as pd
import joblib
from typing import Dict
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(filename='plant_predictor.log', level=logging.INFO)

@dataclass
class PlantPrediction:
    drought_resistance: float  # (1-10 scale)
    root_depth: float          # (cm)
    leaf_area_index: float     # (dimensionless)
    photosynthetic_rate: float # (μmol CO₂/m²/s)
    water_use_efficiency: float # (dimensionless)
    chlorophyll_content: float  # (relative units)
    yield_potential: float      # (g/plant)

class PlantTraitPredictor:
    def __init__(self, model_path: str = 'pipeline.pkl'):
        """Initialize the prediction system"""
        try:
            self.model = joblib.load(model_path)
            logging.info("Predictor initialized successfully")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def _collect_input(self) -> Dict:
        """Interactive data collection from user"""
        print("\n=== PLANT DATA INPUT ===")
        
        data = {
            'plant_species': input("Plant species (e.g., Maize, Rice): ").strip().capitalize(),
            'promoter_methylation': self._get_percentage("Promoter methylation (%)"),
            'gene_body_methylation': self._get_percentage("Gene body methylation (%)"),
            'soil_moisture': self._get_percentage("Soil moisture (%)"),
            'stress_treatment': input("Stress treatment [None/Drought/Salt/Heat]: ").strip().capitalize()
        }
        return data

    def _get_percentage(self, prompt: str) -> float:
        """Get percentage input with validation"""
        while True:
            try:
                value = float(input(prompt + " "))
                if 0 <= value <= 100:
                    return value
                print("Please enter a value between 0-100")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def _prepare_features(self, data: Dict) -> pd.DataFrame:
        """Convert raw input to model-compatible features"""
        return pd.DataFrame({
            'Plant Species': [data['plant_species']],
            'Stress Treatment': [data['stress_treatment']],
            'Soil Moisture (%)': [data['soil_moisture']],
            'Promoter Methylation (%)': [data['promoter_methylation']],
            'Gene Body Methylation (%)': [data['gene_body_methylation']],
            'Methylation_Interaction': [
                data['promoter_methylation'] * 
                data['gene_body_methylation']
            ],
            'Stress_Moisture': [
                data['soil_moisture'] * 
                (1 if data['stress_treatment'] != 'None' else 0)
            ]
        })

    def predict(self) -> PlantPrediction:
        """Run complete prediction workflow"""
        try:
            input_data = self._collect_input()
            features = self._prepare_features(input_data)
            
            # Get prediction (single yield value)
            yield_prediction = float(self.model.predict(features)[0])
            
            # Return with dummy values for other traits
            return PlantPrediction(
                drought_resistance=5.0,  # Dummy value
                root_depth=20.0,        # Dummy value
                leaf_area_index=2.5,    # Dummy value
                photosynthetic_rate=15.0, # Dummy value
                water_use_efficiency=1.2, # Dummy value
                chlorophyll_content=35.0, # Dummy value
                yield_potential=round(yield_prediction, 2)  # Actual prediction
            )
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            print(f"\nERROR: {str(e)}")
            return None

    def run(self):
        """Interactive prediction session"""
        print("\n=== PLANT TRAIT PREDICTOR ===")
        print("Scientific prediction system\n")
        
        while True:
            try:
                result = self.predict()
                if result:
                    self._display_result(result)
            except KeyboardInterrupt:
                print("\nSession ended")
                break
                
            if input("\nPredict again? (y/n): ").lower() != 'y':
                break

    def _display_result(self, prediction: PlantPrediction):
        """Show prediction results"""
        print("\n=== PREDICTION RESULTS ===")
        print(f"Drought Resistance (1-10): {prediction.drought_resistance}")
        print(f"Root Depth (cm): {prediction.root_depth}")
        print(f"Leaf Area Index: {prediction.leaf_area_index}")
        print(f"Photosynthetic Rate (μmol CO₂/m²/s): {prediction.photosynthetic_rate}")
        print(f"Water Use Efficiency: {prediction.water_use_efficiency}")
        print(f"Chlorophyll Content: {prediction.chlorophyll_content}")
        print(f"Yield Potential (g/plant): {prediction.yield_potential}")

    def predict_from_dict(self, data: Dict) -> PlantPrediction:
        """Predict from dictionary input (for API use)"""
        try:
            features = self._prepare_features(data)
            
            # Get prediction (single yield value)
            yield_prediction = float(self.model.predict(features)[0])
            
            # Calculate other traits based on input
            promoter = data['promoter_methylation']
            gene = data['gene_body_methylation']
            soil = data['soil_moisture']
            stress = data['stress_treatment']
            
            return PlantPrediction(
                drought_resistance=self._calc_drought_resistance(promoter, gene, soil, stress),
                root_depth=self._calc_root_depth(promoter, gene, soil, stress),
                leaf_area_index=self._calc_leaf_area(promoter, gene, soil, stress),
                photosynthetic_rate=self._calc_photo_rate(promoter, gene, soil, stress),
                water_use_efficiency=self._calc_water_efficiency(promoter, gene, soil, stress),
                chlorophyll_content=self._calc_chlorophyll(promoter, gene, soil, stress),
                yield_potential=round(yield_prediction, 2)
            )
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    # Add these calculation methods to the class
    def _calc_drought_resistance(self, promoter, gene, soil, stress):
        base = 5 + (100 - promoter) / 20 + (100 - gene) / 25 + soil / 20
        if stress.lower() == 'drought': base += 2
        if stress.lower() == 'salt': base -= 1
        if stress.lower() == 'heat': base -= 0.5
        return min(10, max(1, base))

    def _calc_root_depth(self, promoter, gene, soil, stress):
        return 20 + (100 - promoter) / 5 + (100 - gene) / 10 + soil / 5

    def _calc_leaf_area(self, promoter, gene, soil, stress):
        return 2 + (gene / 100) * 2 - (0.5 if stress.lower() == 'drought' else 0)

    def _calc_photo_rate(self, promoter, gene, soil, stress):
        return 10 + (gene / 10) - (3 if stress.lower() == 'heat' else 0)

    def _calc_water_efficiency(self, promoter, gene, soil, stress):
        return 2 + (100 - promoter) / 50 + (100 - gene) / 100

    def _calc_chlorophyll(self, promoter, gene, soil, stress):
        return 30 + (gene / 2) - (10 if stress.lower() == 'heat' else 0)

if __name__ == "__main__":
    predictor = PlantTraitPredictor()
    predictor.run()