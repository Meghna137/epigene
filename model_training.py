import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import joblib

def clean_data(data):
    """Clean and validate data ranges"""
    # Clip methylation percentages to 0-100 range
    methyl_cols = ['Promoter Methylation (%)', 'Gene Body Methylation (%)',
                 'TE Methylation (%)', 'Global DNA Methylation (%)',
                 'CG Methylation (%)', 'CHG Methylation (%)', 'CHH Methylation (%)']
    
    for col in methyl_cols:
        if col in data.columns:
            data[col] = data[col].clip(0, 100)
    
    # Clean other percentages
    perc_cols = ['Soil Moisture (%)']
    for col in perc_cols:
        if col in data.columns:
            data[col] = data[col].clip(0, 100)
    
    return data

def engineer_features(df):
    """Feature engineering with safe calculations"""
    df['Methylation_Interaction'] = (
        df['Promoter Methylation (%)'].clip(0, 100) * 
        df['Gene Body Methylation (%)'].clip(0, 100))
    
    df['Stress_Moisture'] = (
        df['Soil Moisture (%)'].clip(0, 100) * 
        (df['Stress Treatment'] != 'None').astype(int))
    
    return df

def train_model():
    # Load and clean
    data = pd.read_csv('methylation_dataset.csv')
    data = clean_data(data)
    
    # Drop rows where target is missing
    target_col = 'Yield (g/plant)'
    data = data.dropna(subset=[target_col])
    
    # Feature engineering
    data = engineer_features(data)
    
    # Define features
    numerical_cols = ['Soil Moisture (%)', 'Promoter Methylation (%)', 
                     'Gene Body Methylation (%)', 'Methylation_Interaction',
                     'Stress_Moisture']
    categorical_cols = ['Plant Species', 'Stress Treatment']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(mutual_info_regression, k=10)),
        ('pca', PCA(n_components=0.95)),
        ('model', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    X = data[numerical_cols + categorical_cols]
    y = data[target_col]
    pipeline.fit(X, y)
    
    # Evaluate
    y_pred = pipeline.predict(X)
    print(f"\nTraining R2 Score: {r2_score(y, y_pred):.2f}")
    print(f"Feature Importances: {pipeline.named_steps['model'].feature_importances_}")
    
    # Save
    joblib.dump(pipeline, 'pipeline.pkl')
    print("\nPipeline saved successfully!")

if __name__ == "__main__":
    train_model()