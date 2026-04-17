import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# =========================
# LOAD & TRAIN MODEL (runs once)
# =========================
@st.cache_resource
def load_model():
    df = pd.read_csv(r"C:\Users\arpit\OneDrive\Desktop\gen ai\Hostel Management (Responses) - Form Responses 1.csv")

    df = df.rename(columns={
        'Please describe your complaint in detail  ': 'text',
        'Select the category that best matches your complaint  ': 'category'
    })

    df = df[['text', 'category']].dropna()
    df['text'] = df['text'].str.lower()

    # Clean categories
    def clean_category(cat):
        cat = str(cat).lower()
        if "fan" in cat or "light" in cat or "ac" in cat or "electrical" in cat:
            return "Electrical"
        elif "wifi" in cat or "internet" in cat:
            return "Internet"
        elif "mess" in cat or "food" in cat:
            return "Food"
        elif "clean" in cat or "hygiene" in cat:
            return "Cleanliness"
        elif "plumbing" in cat or "water" in cat:
            return "Plumbing"
        elif "maintenance" in cat or "facility" in cat:
            return "Maintenance"
        elif "security" in cat:
            return "Security"
        else:
            return "Other"

    df['category'] = df['category'].apply(clean_category)

    # Remove weak class
    df = df[df['category'] != "Other"]

    # Encode
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['category'])

    # Tokenize
    tokenizer = Tokenizer(num_words=6000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])

    X_seq = tokenizer.texts_to_sequences(df['text'])
    X_pad = pad_sequences(X_seq, maxlen=60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42
    )

    # Model
    model = Sequential([
        Embedding(6000, 64),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=0)

    return model, tokenizer, encoder


model, tokenizer, encoder = load_model()

# =========================
# PREDICTION FUNCTION
# =========================
def predict(text):
    seq = tokenizer.texts_to_sequences([text.lower()])
    pad = pad_sequences(seq, maxlen=60)
    
    pred = model.predict(pad)
    return encoder.inverse_transform([np.argmax(pred)])[0]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Hostel Complaint AI", page_icon="🏨")

st.title("🏨 Hostel Complaint Management System")
st.write("Enter your complaint and let AI classify it instantly")

# Input fields
name = st.text_input("👤 Your Name")
room = st.text_input("🏠 Room Number")
complaint = st.text_area("📝 Enter your complaint")

# Button
if st.button("Submit Complaint"):
    if complaint.strip() == "":
        st.warning("Please enter a complaint")
    else:
        category = predict(complaint)

        st.success("✅ Complaint Registered Successfully!")

        st.subheader("🔍 AI Prediction")
        st.write(f"**Category:** {category}")

        st.info(f"📌 Assigned to: {category} Department")