# 🏨 AI Hostel Complaint Classification System (LSTM Based)

A deep learning project that automatically classifies hostel complaints and predicts their urgency using an LSTM-based Recurrent Neural Network.

---

## 🚀 Project Overview

This system takes a student's complaint in natural language and predicts:

* 📌 **Complaint Category**
  (Electrical, Plumbing, Internet, Cleanliness, Furniture, etc.)

* ⚡ **Priority Level**
  (Low, Medium, Urgent)

The model is designed as a **multi-output neural network**, meaning it predicts both category and priority simultaneously.

---

## 🧠 Model Architecture

The core of this project is an LSTM-based RNN built using TensorFlow/Keras.

### Architecture Flow:

1. **Input Layer**
2. **Embedding Layer**

   * Converts words into dense vector representations
3. **LSTM Layer**

   * Captures sequence and context from complaint text
4. **Dropout Layer**

   * Prevents overfitting
5. **Two Output Layers**:

   * Category (Softmax)
   * Priority (Softmax)

---

## 🔄 Workflow Pipeline

### 1. Data Collection

* Dataset collected via Google Forms
* Real hostel complaints provided by users

### 2. Data Preprocessing

* Convert text to lowercase
* Remove punctuation
* Tokenization
* Sequence padding

### 3. Text Encoding

* Tokenizer converts text → sequences
* Sequences padded to fixed length

### 4. Label Encoding

* Categories and priorities encoded using LabelEncoder

### 5. Model Training

* Loss: Sparse Categorical Crossentropy
* Optimizer: Adam
* Multi-output training

---

## 📂 Dataset Details

The dataset contains:

* `complaint_text` → Description of issue
* `category` → Type of complaint
* `priority` → Urgency level

Example:

```text
"There is water leakage in my bathroom" → Plumbing, High
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/ai-hostel-complaint-system.git
cd ai-hostel-complaint-system
pip install -r requirements.txt
```

Run the model:

```bash
python mm.py
```

---

## 🧪 Example Prediction

```python
sample = "Fan is not working in my room"
print(predict_complaint(sample))
```

### Output:

```text
('Electrical', 'Urgent')
```

---

## 📊 Model Highlights

* ✅ LSTM-based sequence learning
* ✅ Multi-output classification (Category + Priority)
* ✅ Handles natural language input
* ✅ End-to-end pipeline from raw text → prediction

---

## ⚠️ Challenges Faced

* Small dataset size
* Class imbalance in complaint categories
* Misclassification due to limited examples
  (e.g., plumbing issues predicted as electrical)

---

## 🔧 Improvements Implemented

* Data cleaning and normalization
* Label standardization
* Dropout layer to reduce overfitting
* Increased training epochs

---

## 🚀 Future Enhancements

* 🔹 Increase dataset size for better accuracy
* 🔹 Use advanced models like BERT
* 🔹 Build Streamlit-based UI
* 🔹 Deploy as a web application
* 🔹 Add auto-response system

---

## 💡 Key Learnings

* Natural Language Processing using deep learning
* Working with LSTM networks
* Multi-output model design
* Handling real-world noisy datasets
* Debugging model errors (like multi-metric issue)

---

## 👩‍💻 Author

**Arpita Dubey**

---

⭐ If you find this project useful, consider starring the repository!
