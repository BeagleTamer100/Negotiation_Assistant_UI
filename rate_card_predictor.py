import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb

class RateCardPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_trained = False
        self.current_year = 2025  # Set current year to 2025
        
    def load_data(self, file_path):
        """Load and preprocess the rate card data"""
        self.data = pd.read_excel(file_path)
        # Remove any rows with NaN in Rate_Card
        self.data = self.data.dropna(subset=['Rate_Card'])
        # Fill NaN values in other columns with appropriate defaults
        self.data = self.data.fillna({
            'Delta PPI': 0,
            'Delta GDP': 0,
            'Delta Electricity': 0,
            'Delta Labor': 0,
            'Inflation': 0
        })
        return self.data
        
    def preprocess_data(self):
        """Create preprocessing pipeline for the data"""
        # Define categorical and numerical columns
        categorical_features = ['Supplier', 'Roles', 'Experience', 'Currency', 'Country']
        numerical_features = ['Year', 'Delta PPI', 'Delta GDP', 'Delta Electricity', 'Delta Labor', 'Inflation']
        
        # Create preprocessing steps for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
        
    def train_model(self, data=None):
        """Train the XGBoost model"""
        if data is not None:
            self.data = data
            
        # Prepare features and target
        X = self.data.drop(['Rate_Card'], axis=1)  # Keep Year in features
        y = self.data['Rate_Card']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit preprocessor if not already created
        if self.preprocessor is None:
            self.preprocess_data()
        
        # Create and train the model pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate and return test score
        test_score = self.model.score(X_test, y_test)
        return test_score
        
    def predict_rate(self, input_data):
        """Predict rate card value for new input"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before making predictions")
            
        # Convert input_data to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        # Set Year to 2025 if not present
        if 'Year' not in input_data.columns:
            input_data['Year'] = 2025
        else:
            input_data['Year'] = 2025  # Override any provided year with 2025
            
        # Make prediction
        prediction = self.model.predict(input_data)
        return prediction[0]  # Return single prediction value

    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error for the model"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before calculating MAPE")
            
        # Get test predictions
        X = self.data.drop(['Rate_Card'], axis=1)
        y = self.data['Rate_Card']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = self.model.predict(X_test)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        return round(mape, 2)
        
    def predict_all_china_positions(self):
        """Predict rates for all positions in China"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained before making predictions")
            
        # Get unique combinations of Supplier, Roles, and Experience for China
        china_data = self.data[self.data['Country'] == 'China']
        unique_combinations = china_data[['Supplier', 'Roles', 'Experience']].drop_duplicates()
        
        # Create input data for all combinations
        predictions = []
        for _, row in unique_combinations.iterrows():
            input_data = {
                'Supplier': row['Supplier'],
                'Roles': row['Roles'],
                'Experience': row['Experience'],
                'Country': 'China',
                'Year': 2025,
                'Delta PPI': -0.070,
                'Delta GDP': 425,
                'Delta Electricity': 0.048,
                'Delta Labor': 8000,
                'Inflation': 0.0043
            }
            
            predicted_rate = self.predict_rate(input_data)
            predictions.append({
                'supplier': row['Supplier'],
                'role': row['Roles'],
                'experience': row['Experience'],
                'predicted_rate': round(predicted_rate, 2)
            })
            
        return sorted(predictions, key=lambda x: (x['supplier'], x['role'], x['experience']))

# Example usage:
# predictor = RateCardPredictor()
# data = predictor.load_data('Rate_Cards_20.22.24.V2.xlsx')
# score = predictor.train_model()
# 
# new_input = {
#     'Supplier': 'Supplier 1',
#     'Roles': 'Developer',
#     'Experience': 'Senior',
#     'Currency': 'USD',
#     'Country': 'China',
#     'Year': 2024,
#     'Delta PPI': -0.070,
#     'Delta GDP': 425,
#     'Delta Electricity': 0.048,
#     'Delta Labor': 8000,
#     'Inflation': 0.0043
# }
# predicted_rate = predictor.predict_rate(new_input) 