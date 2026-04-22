import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

app = Flask(__name__)

# Constants
MAX_LEN = 60

# Load Model and Artifacts
model = None
tokenizer = None
encoder = None

def load_artifacts():
    global model, tokenizer, encoder
    try:
        if os.path.exists("model.h5"):
            model = load_model("model.h5")
            print("Model loaded successfully.")
        if os.path.exists("tokenizer.pkl"):
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully.")
        if os.path.exists("encoder.pkl"):
            with open("encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            print("Encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

# In-memory storage for complaints (for demo purposes)
complaints_db = [
    {
        "id": "CMS-9842",
        "text": "Faulty Wiring Issue",
        "category": "Electrical",
        "room": "203",
        "floor": "2",
        "urgency": "High",
        "status": "In Progress",
        "timestamp": "Oct 24, 2023",
        "update_time": "2 hours ago"
    },
    {
        "id": "CMS-9843",
        "text": "Water leakage in washroom",
        "category": "Plumbing",
        "room": "105",
        "floor": "1",
        "urgency": "Medium",
        "status": "Resolved",
        "timestamp": "Oct 22, 2023",
        "update_time": "Updated 2h ago"
    }
]

def extract_info(text):
    info = {
        "room": "N/A",
        "floor": "N/A",
        "urgency": "Low"
    }
    
    # Room Number: "room 203", "rm 203", "203 room"
    room_match = re.search(r'(?:room|rm|unit)\s*(\d+)', text, re.I)
    if room_match:
        info["room"] = room_match.group(1)
    
    # Floor Number: "2nd floor", "floor 2", "second floor"
    floor_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*floor', text, re.I)
    if not floor_match:
        floor_match = re.search(r'floor\s*(\d+)', text, re.I)
    if floor_match:
        info["floor"] = floor_match.group(1)
        
    # Urgency Level
    high_urgency = ["urgent", "emergency", "asap", "immediately", "critical", "blocking", "broken", "danger"]
    medium_urgency = ["priority", "soon", "needed", "fast", "leak", "moderate"]
    
    text_lower = text.lower()
    for word in high_urgency:
        if word in text_lower:
            info["urgency"] = "High"
            break
    
    if info["urgency"] == "Low":
        for word in medium_urgency:
            if word in text_lower:
                info["urgency"] = "Medium"
                break
                
    return info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None or encoder is None:
        load_artifacts()
    
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 500
        
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        # ML Prediction
        seq = tokenizer.texts_to_sequences([text.lower().strip()])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prob = model.predict(pad, verbose=0)[0]
        category_idx = np.argmax(prob)
        category = encoder.classes_[category_idx]
        
        # Information Extraction
        info = extract_info(text)
        
        now = datetime.now()
        timestamp = now.strftime("%b %d, %Y")
        
        response = {
            "id": f"CMS-{9844 + len(complaints_db)}",
            "text": text,
            "category": category,
            "room": info["room"],
            "floor": info["floor"],
            "urgency": info["urgency"],
            "status": "Pending",
            "timestamp": timestamp,
            "update_time": "Just now"
        }
        
        complaints_db.insert(0, response) # Add to start of list
        return jsonify(response)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify(complaints_db)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
