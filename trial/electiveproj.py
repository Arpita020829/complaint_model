import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\arpit\OneDrive\Desktop\gen ai\Hostel Management (Responses) - Form Responses 1.csv")

df = df.rename(columns={
    'Please describe your complaint in detail  ': 'text',
    'Select the category that best matches your complaint  ': 'category'
})

df = df[['text', 'category']].dropna()
df['text'] = df['text'].str.lower()

# =========================
# 2. BETTER CATEGORY CLEANING
# =========================
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

# REMOVE "Other" (VERY IMPORTANT FIX)
df = df[df['category'] != "Other"]

# =========================
# 3. ENCODE LABELS
# =========================
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])

# =========================
# 4. TOKENIZATION
# =========================
tokenizer = Tokenizer(num_words=6000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=60)

# =========================
# 5. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# =========================
# 6. MODEL (IMPROVED)
# =========================
model = Sequential([
    Embedding(6000, 64),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# =========================
# 7. TRAIN (IMPORTANT CHANGE)
# =========================
model.fit(X_train, y_train, epochs=10, batch_size=8)

# =========================
# 8. PREDICTION
# =========================
def predict(text):
    seq = tokenizer.texts_to_sequences([text.lower()])
    pad = pad_sequences(seq, maxlen=60)
    
    pred = model.predict(pad)
    return encoder.inverse_transform([np.argmax(pred)])[0]

# =========================
# 9. INTERACTIVE LOOP
# =========================
print("\n🤖 Improved RNN Ready!")

while True:
    text = input("\nEnter complaint (type 'exit'): ")
    
    if text.lower() == "exit":
        break
    
    print("Predicted Category:", predict(text))
