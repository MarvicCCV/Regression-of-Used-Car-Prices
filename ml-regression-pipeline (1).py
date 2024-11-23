# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Initial Exploration
def load_and_explore_data(train_path, test_path):
    # Load both datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print("Training Dataset Info:")
    print(df_train.info())
    print("\nTraining Data Missing Values:")
    print(df_train.isnull().sum())
    
    print("\nTest Dataset Info:")
    print(df_test.info())
    print("\nTest Data Missing Values:")
    print(df_test.isnull().sum())
    
    # Check for any differences in columns between train and test
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    if train_cols != test_cols:
        print("\nColumn differences between train and test:")
        print("Columns only in train:", train_cols - test_cols)
        print("Columns only in test:", test_cols - train_cols)
    
    return df_train, df_test

# 2. Data Analysis and Visualization
def analyze_data(df_train):
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df_train['price'], bins=50)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price')
    plt.show()
    
    # Correlation analysis for numeric columns
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_train[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.show()

# 3. Data Preprocessing
def preprocess_data(df_train, df_test):
    # Separate features and target
    X_train = df_train.drop(['price', 'id'], axis=1)
    y_train = df_train['price']
    X_test = df_test.drop(['id'], axis=1)  # Assuming test set doesn't have 'price'
    
    # Define numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
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
    
    return X_train, y_train, X_test, preprocessor

# 4. Model Development and Training
def train_model(X_train, y_train, preprocessor):
    # Create model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model

# 5. Model Evaluation
def evaluate_model(model, X_train, y_train, X_test=None, y_test=None):
    # Training predictions
    y_train_pred = model.predict(X_train)
    
    # Calculate training metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("Training Metrics:")
    print(f"RMSE: ${train_rmse:,.2f}")
    print(f"MAE: ${train_mae:,.2f}")
    print(f"R2 Score: {train_r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Car Prices (Training Data)')
    plt.tight_layout()
    plt.show()
    
    # If test labels are available, evaluate on test set
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\nTest Metrics:")
        print(f"RMSE: ${test_rmse:,.2f}")
        print(f"R2 Score: {test_r2:.4f}")

# 6. Feature Importance Analysis
def analyze_feature_importance(model, X_train):
    feature_names = (X_train.columns.tolist() + 
                    model.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names(X_train.select_dtypes(include=['object']).columns).tolist())
    
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# 7. Generate Predictions for Test Set
def generate_predictions(model, X_test, test_df, output_path):
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'price': predictions
    })
    
    # Save predictions
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return submission

# 8. Main Execution
def main():
    # Load data
    df_train, df_test = load_and_explore_data('data/train.csv', 'data/test.csv')
    
    # Analyze training data
    analyze_data(df_train)
    
    # Preprocess data
    X_train, y_train, X_test, preprocessor = preprocess_data(df_train, df_test)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model on training data
    evaluate_model(model, X_train, y_train)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, X_train)
    
    # Generate and save predictions
    predictions = generate_predictions(model, X_test, df_test, 'predictions.csv')

if __name__ == "__main__":
    main()
