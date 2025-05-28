from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import os
app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client.sentimentDB
collection = db.reviews
classifier = pipeline("sentiment-analysis")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json["text"]
    result = classifier(text)[0]
    sentiment = result["label"]
    score = float(result["score"])
    collection.insert_one({
        "text": text,
        "sentiment": sentiment,
        "score": score,
        "timestamp": datetime.utcnow()
    })
    return jsonify({"sentiment": sentiment, "score": score})
@app.route("/history")
def history():
    data = list(collection.find().sort("timestamp", -1).limit(20))
    return jsonify([{"sentiment": d["sentiment"], "score": d["score"]} for d in data])
if __name__ == "__main__":
    app.run(debug=True)