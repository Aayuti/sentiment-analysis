import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data_amazon.csv')

# Preprocess data
df = df.dropna(subset=['Summary', 'Score'])
df['Summary'] = df['Summary'].astype(str)
texts = df['Summary'].values
labels = df['Score'].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test_vec)

# Print classification report
report = classification_report(y_test, y_pred)
print(report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Naive Bayes Model')
plt.show()


# print(classification_report(y_test, y_pred))
