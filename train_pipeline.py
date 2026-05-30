import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train():
    # 1. Load dataset
    dataset_path = "adult 3.csv"
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
        
    data = pd.read_csv(dataset_path)
    print(f"Original data shape: {data.shape}")
    
    # 2. Filter age outlier (keeping between 17 and 75)
    print("Filtering age between 17 and 75...")
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]
    print(f"Data shape after age filtering: {data.shape}")
    
    # 3. Clean categorical columns by replacing '?' with specific classes
    print("Replacing '?' in occupation and workclass...")
    data['occupation'] = data['occupation'].replace({"?": "others"})
    data['workclass'] = data['workclass'].replace({"?": "NotListed"})
    
    # 4. Filter out workclass levels 'Without-pay' and 'Never-worked'
    print("Filtering out Without-pay and Never-worked workclass...")
    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    
    # 5. Filter out education levels '5th-6th', '1st-4th', and 'Preschool'
    print("Filtering out minor education levels...")
    data = data[data['education'] != '5th-6th']
    data = data[data['education'] != '1st-4th']
    data = data[data['education'] != 'Preschool']
    
    # 6. Drop redundant 'education' column (since 'educational-num' represents it numerically)
    print("Dropping redundant education column...")
    data = data.drop(columns=['education'])
    print(f"Data shape after cleaning: {data.shape}")
    
    # 7. Setup categorical columns and fit individual LabelEncoders
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    encoders = {}
    
    print("Fitting LabelEncoders...")
    for col in categorical_cols:
        encoder = LabelEncoder()
        # Fit on all possible values of the column in the cleaned dataset
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder
        print(f"Encoded '{col}' with classes: {list(encoder.classes_)}")
        
    # 8. Separate features and target
    X = data.drop(columns=['income'])
    y = data['income']
    
    # Keep the feature columns order for reference
    feature_cols = list(X.columns)
    print(f"Feature columns: {feature_cols}")
    
    # 9. Fit MinMaxScaler on features
    print("Fitting MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 10. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 11. Train GradientBoostingClassifier (which was the best model in the notebook)
    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 12. Evaluate model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Gradient Boosting Model accuracy on test set: {acc:.4f}")
    
    # 13. Save artifacts
    print("Saving pipeline artifacts...")
    joblib.dump(model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    # Save the encoders and the feature columns list so we can reconstruct inputs accurately
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, "encoders.pkl")
    
    print("Artifacts successfully saved:")
    print(" - best_model.pkl")
    print(" - scaler.pkl")
    print(" - encoders.pkl")

if __name__ == "__main__":
    train()
