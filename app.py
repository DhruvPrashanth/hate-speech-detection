from flask import Flask, request, render_template
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils import clean_text


app = Flask(__name__)

model = DistilBertForSequenceClassification.from_pretrained("model/")
tokenizer = DistilBertTokenizer.from_pretrained("model/")

labels = ["Hate Speech", "Offensive", "Neutral"]

def predict(text):
    text = clean_text(text)

    if not text:   # handle empty after cleaning
        return None, None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    probabilities = {
        "Hate Speech": round(probs[0].item() * 100, 2),
        "Offensive": round(probs[1].item() * 100, 2),
        "Neutral": round(probs[2].item() * 100, 2),
    }

    predicted_label = labels[torch.argmax(probs).item()]

    return predicted_label, probabilities

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probabilities = None

    if request.method == "POST":
        text = request.form["text"].strip()

        if text:  # only run if not empty
            prediction, probabilities = predict(text)
        else:
            prediction = None
            probabilities = None

    return render_template("index.html", prediction=prediction, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)