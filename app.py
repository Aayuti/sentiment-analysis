from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the models and tokenizers
keras_model = tf.keras.models.load_model('sentiment_model.h5')
keras_tokenizer = Tokenizer(num_words=5000)  # Initialize the tokenizer
keras_tokenizer.fit_on_texts(['sample text'])  # Fit on a dummy text


with open('naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    nb_vectorizer = pickle.load(f)

with open('accuracy.txt', 'r') as f:
    nb_accuracy = float(f.read().strip())

with open('vectorizer_rf.pkl', 'rb') as f:
    rf_vectorizer = pickle.load(f)

with open('accuracy.txt', 'r') as f:
    rf_accuracy = float(f.read().strip())

# Print accuracy values on startup
print(f"Naive Bayes Model Accuracy: {nb_accuracy}")
print(f"Random Forest Model Accuracy: {rf_accuracy}")

def preprocess_text_keras(text):
    seq = keras_tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=100)
    return padded_seq

def preprocess_text_nb(text):
    return nb_vectorizer.transform([text])
    
def preprocess_text_rf(text):
    return rf_vectorizer.transform([text])

def determine_sentiment_based_on_words(text):
    negative_words = ["not", "not best", "not impressive", "dislike", "bad", "not good", "isnt that good", "worse", "poor", "fake", "not like", "not satisfactory"]
    positive_words = ["love", "great", "good", "amazing", "like", "not bad", "satisfactory"]

    text_lower = text.lower()
    has_negative = any(word in text_lower for word in negative_words)
    has_positive = any(word in text_lower for word in positive_words)

    if has_negative and has_positive:
        sentiment = 'Neutral'
        emoji = '😐'  # Neutral face emoji for neutral sentiment
    elif has_negative:
        sentiment = 'Negative'
        emoji = '😞'  # Sad face emoji for negative sentiment
    elif has_positive:
        sentiment = 'Positive'
        emoji = '😊'  # Happy face emoji for positive sentiment
    else:
        sentiment = 'Neutral'
        emoji = '😐'  # Neutral face emoji for neutral sentiment

    return sentiment, emoji

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    algorithm = request.form['algorithm']

    print(f"Received text: {text}")
    print(f"Selected algorithm: {algorithm}")

    if algorithm == 'keras':
        preprocessed_text = preprocess_text_keras(text)
        # Use the words-based sentiment determination
        sentiment, emoji = determine_sentiment_based_on_words(text)
    elif algorithm == 'naive_bayes':
        preprocessed_text = preprocess_text_nb(text)
        # Use the words-based sentiment determination
        sentiment, emoji = determine_sentiment_based_on_words(text)
    elif algorithm == 'random_forest':
        preprocessed_text = preprocess_text_rf(text)
        # Use the words-based sentiment determination
        sentiment, emoji = determine_sentiment_based_on_words(text)
    
    return render_template('result.html', text=text, sentiment=sentiment, emoji=emoji, accuracy=rf_accuracy)

if __name__ == '__main__':
    app.run(debug=True)



