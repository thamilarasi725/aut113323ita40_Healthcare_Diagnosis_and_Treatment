import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (simplified)
data = {
    'Fever': [1, 0, 1, 0, 1],
    'Cough': [1, 1, 0, 0, 1],
    'Fatigue': [1, 0, 1, 1, 0],
    'Headache': [0, 0, 1, 1, 1],
    'Diagnosis': ['Flu', 'Cold', 'Migraine', 'Stress', 'Flu']
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Fever', 'Cough', 'Fatigue', 'Headache']]
y = df['Diagnosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Diagnosis prediction function
def predict_diagnosis(symptoms):
    input_data = pd.DataFrame([symptoms])
    prediction = model.predict(input_data)[0]
    treatments = {
        'Flu': 'Rest, fluids, antiviral drugs',
        'Cold': 'Rest, hydration, OTC meds',
        'Migraine': 'Pain relievers, dark room, avoid triggers',
        'Stress': 'Relaxation, exercise, counseling'
    }
    return prediction, treatments.get(prediction, "Consult a doctor")

# Example usage
symptoms_input = {'Fever': 1, 'Cough': 1, 'Fatigue': 0, 'Headache': 0}
diagnosis, treatment = predict_diagnosis(symptoms_input)
print(f"Diagnosis: {diagnosis}\nTreatment: {treatment}")