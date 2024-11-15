import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# data = ...
data = pd.read_csv('data_amazon.csv')

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Summary'])
y = data['Score']  # Assuming you have labels for training

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Save the model and vectorizer
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('vectorizer_rf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Test accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Print classification report
report = classification_report(y_test, y_pred)
print(report)
