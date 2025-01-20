import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and prepare the data
df = pd.read_csv('car evaluation(2).csv')
le = {}
for column in df.columns:
    le[column] = LabelEncoder()
    df[column] = le[column].fit_transform(df[column])

# Train the model
X = df.drop('evaluation level', axis=1)
y = df['evaluation level']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the model and label encoders
joblib.dump(model, 'car_evaluation_model(2).model')
joblib.dump(le, 'label_encoders.pkl')
