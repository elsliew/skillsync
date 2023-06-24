import os
from flask import Flask, render_template
from flask import request
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

access_token='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'
model = AutoModelForSequenceClassification.from_pretrained("elsliew/autotrain-skillsync2-69166137722", use_auth_token=access_token)
tokenizer = AutoTokenizer.from_pretrained("elsliew/autotrain-skillsync2-69166137722", use_auth_token=access_token)

def analyze_sentiment(input_data):
    tokens = tokenizer.encode(input_data, return_tensors='pt')
    results = model(tokens)
    logits = results.logits.squeeze(0)
    probabilities = torch.softmax(logits, dim=0)
    sentiment_score = torch.dot(torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), probabilities)
    confidence_score = max(probabilities).item()
    skillsync = round(sentiment_score.item(), 2)
    return skillsync

app = Flask(__name__)

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    input_text = data['text']
    sentiment_score = analyze_sentiment(input_text)
    response = {
        'sentiment_score': sentiment_score
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
