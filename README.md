# data-science-project-
[5/8, 12:01 PM] Arthi Clg Frd: import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('traffic_accidents.csv')

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data['weather_condition'] = imputer.fit_transform(data[['weather_condition']])
data['traffic_density'] = imputer.fit_transform(data[['traffic_density']])

# Convert timestamp to datetime and extract useful features
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Encode categorical variables (weather, traffic_density)
le_weather = LabelEncoder()
data['weather_condition'] = le_weather.fit_transform(data['weather_condition'])

le_traffic = LabelEncoder()
data['traffic_density'] = le_traffic.fit_transform(data['traffic_density'])

# Define features and target
X = data[['hour', 'day_of_week', 'weather_condition', 'traffic_density', 'vehicle_speed']]
y = data['accident']
[5/8, 12:01 PM] Arthi Clg Frd: from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
[5/8, 12:01 PM] Arthi Clg Frd: # Sample input: predict for a scenario (e.g., rush hour, rainy weather)
sample_data = pd.DataFrame({
    'hour': [8],  # 8 AM
    'day_of_week': [1],  # Monday
    'weather_condition': [le_weather.transform(['rainy'])[0]],  # rainy weather
    'traffic_density': [le_traffic.transform(['high'])[0]],  # high traffic density
    'vehicle_speed': [30]  # 30 km/h
})

# Predict accident occurrence
accident_pred = model.predict(sample_data)
print("Accident Prediction:", "Accident" if accident_pred[0] == 1 else "No Accident")
[5/8, 12:01 PM] Arthi Clg Frd: import matplotlib.pyplot as plt
import seaborn as sns

# Plotting accident frequency over hours of the day
sns.countplot(data=data, x='hour', hue='accident')
plt.title('Accident Frequency by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# Plotting accidents by weather condition
sns.countplot(data=data, x='weather_condition', hue='accident')
plt.title('Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.show()
