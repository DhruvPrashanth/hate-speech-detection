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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    print("Probabilities:", probs)

    return labels[torch.argmax(probs).item()]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        prediction = predict(text)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)