# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Initial Exploration
def load_and_explore_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic information about the dataset
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())
    
    return df

# 2. Data Preprocessing
def preprocess_data(df):
    # Separate features and target
    X = df.drop(['price', 'id'], axis=1)  # Assuming 'price' is target and 'id' is not needed
    y = df['price']
    
    # Define numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

# 3. Model Development and Training
def train_model(X, y, preprocessor):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 4, 5],
        'regressor__learning_rate': [0.01, 0.1]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    return grid_search, X_train, X_test, y_train, y_test

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Car Prices')
    plt.tight_layout()
    plt.show()

# 5. Feature Importance Analysis
def analyze_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.show()

# 6. Main Execution
def main():
    # Load and explore data
    df = load_and_explore_data('data/train.csv')
    
    # Preprocess data
    X, y, preprocessor = preprocess_data(df)
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y, preprocessor)
    print("\nBest Parameters:", model.best_params_)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(model.best_estimator_.named_steps['regressor'], X)

if __name__ == "__main__":
    main()
