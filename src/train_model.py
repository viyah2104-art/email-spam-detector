import pandas as pd
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/spam.csv")

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

df['text'] = df['text'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save files
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")