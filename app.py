# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from predictor import PlantTraitPredictor
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load your trained model
try:
    predictor = PlantTraitPredictor('pipeline.pkl')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    predictor = None

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        
        # Prepare input data structure
        input_data = {
            'plant_species': data['species'],
            'promoter_methylation': float(data['promoter_methylation']),
            'gene_body_methylation': float(data['gene_methylation']),
            'soil_moisture': float(data['soil']),
            'stress_treatment': data['stress']
        }
        
        # Get prediction
        prediction = predictor.predict_from_dict(input_data)
        
        return jsonify({
            'drought_resistance': prediction.drought_resistance,
            'root_depth': prediction.root_depth,
            'leaf_area_index': prediction.leaf_area_index,
            'photosynthetic_rate': prediction.photosynthetic_rate,
            'water_use_efficiency': prediction.water_use_efficiency,
            'chlorophyll_content': prediction.chlorophyll_content,
            'yield_potential': prediction.yield_potential
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)