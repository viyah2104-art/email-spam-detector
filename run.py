from src.train_model import *
from src.predictor import predict_email

print("📩 Email Spam Detector\n")

while True:
    text = input("Enter email text (or type exit): ")

    if text.lower() == "exit":
        break

    result = predict_email(text)

    if result == "Spam":
        print("🚫 Spam Email\n")
    else:
        print("✅ Not Spam\n")