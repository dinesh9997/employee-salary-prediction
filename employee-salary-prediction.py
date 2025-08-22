import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('adult 3.csv')

# Data Exploration
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nIncome Distribution:")
print(data['income'].value_counts())

# Data Preprocessing
# Replace '?' with NaN
data = data.replace('?', np.nan)

# Handle missing values
for col in ['workclass', 'occupation', 'native-country']:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Convert income to binary (0 for <=50K, 1 for >50K)
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

# Encode categorical variables
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                   'relationship', 'race', 'gender', 'native-country']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Feature selection - drop fnlwgt as it's a weighting factor, not a feature
data = data.drop('fnlwgt', axis=1)

# Split data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
numerical_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualization
plt.figure(figsize=(15, 10))

# 1. Income Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='income', data=data)
plt.title('Income Distribution')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Count')

# 2. Age vs Income
plt.subplot(2, 2, 2)
sns.boxplot(x='income', y='age', data=data)
plt.title('Age Distribution by Income')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Age')

# 3. Education vs Income
plt.subplot(2, 2, 3)
sns.boxplot(x='income', y='educational-num', data=data)
plt.title('Education Level by Income')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Education Level')

# 4. Feature Importance
plt.subplot(2, 2, 4)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K', '>50K'], 
            yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Additional Analysis: Hours per week vs Income
plt.figure(figsize=(8, 6))
sns.boxplot(x='income', y='hours-per-week', data=data)
plt.title('Working Hours per Week by Income')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Hours per Week')
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='income', y='capital-gain', data=data)
plt.title('Capital Gain by Income')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Capital Gain')

plt.subplot(1, 2, 2)
sns.boxplot(x='income', y='capital-loss', data=data)
plt.title('Capital Loss by Income')
plt.xlabel('Income (0=<=50K, 1=>50K)')
plt.ylabel('Capital Loss')
plt.tight_layout()
plt.show()


print("\nKey Findings:")
print("1. The dataset is imbalanced with more people earning <=50K than >50K")
print("2. Age, education level, and working hours are positively correlated with higher income")
print("3. Capital gains are a strong indicator of higher income")
print("4. The model achieved good accuracy in predicting income levels")
print("5. The most important features for prediction are age, education level, and capital gain")