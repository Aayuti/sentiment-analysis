import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('data_amazon.csv')

#negation handling
def handle_negation(text):
    # Check if text is a valid string before processing
    if isinstance(text, str):
        # Replace "not like" with "dislike" as a simple example of negation handling
        text = re.sub(r'\bnot like\b', 'dislike', text)
        # General pattern to handle other "not <word>" scenarios
        text = re.sub(r'not\s+([a-zA-Z]+)', r'not_\1', text)
    return text


df['Summary'] = df['Summary'].apply(handle_negation)

# Preprocess data
# Drop rows with missing values in 'summary' and 'score' columns
df = df.dropna(subset=['Summary', 'Score'])

# Convert the 'summary' column to string type to prevent AttributeError
df['Summary'] = df['Summary'].astype(str)

# Get texts and labels
texts = df['Summary'].values
labels = df['Score'].values


# Convert labels to categorical
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)  # Convert to one-hot encoding

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)  # Fit tokenizer on the text data
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

