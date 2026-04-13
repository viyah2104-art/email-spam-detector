import pickle

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_email(text):
    data = vectorizer.transform([text])
    result = model.predict(data)[0]
    return "Spam" if result == 1 else "Not Spam"
