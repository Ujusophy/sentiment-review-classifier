# Sentiment Review Classifier

### ✅ Binary Sentiment Classifier using DistilBERT for E-commerce Product Reviews

---

## 📌 Project Summary

This project demonstrates how to fine-tune a **transformer-based NLP model (DistilBERT)** for **sentiment classification** on short product reviews. It aims to distinguish between **positive (1)** and **negative (0)** reviews, a valuable task in **e-commerce**, **customer feedback analysis**, and **automated review moderation systems**.

The pipeline includes:
- 📂 Dataset preparation from UCI Sentiment Labelled Sentences dataset
- 🤖 Fine-tuning `distilbert-base-uncased` using Hugging Face Transformers
- 🧪 Inference scripts for CLI and API access
- 🌐 Option to deploy via FastAPI for production use

---

## 📊 Example Use Case

Imagine running a marketplace or online store—this tool helps you:
- Automatically **flag bad reviews** for moderation
- Monitor **customer sentiment over time**
- Build smart **recommendation or support systems**

---

## 📁 Project Structure

```bash
sentiment-review-classifier/
├── sentiment_dataset                             
├── prepare_dataset.py            
├── train_model.py             
│── evaluate_model.py                                       
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📚 Dataset

- **Source**: [UCI Sentiment Labelled Sentences Data Set]([https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip))
- **Files**: 
  - `amazon_cells_labelled.txt`
  - `imdb_labelled.txt`
  - `yelp_labelled.txt`
- **Structure**:
  - Each line contains a short sentence and a binary label (`0 = negative`, `1 = positive`)
  - ~3,000 total examples

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Ujusophy/sentiment-review-classifier.git
cd sentiment-review-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
```bash
python dataset/prepare_dataset.py
```

### 4. Train the model
```bash
python model/train_model.py
```

### 5. Run inference (CLI)
```bash
python inference/predict.py
```

---

## 🔍 Sample CLI Output

```
Enter a review (or 'exit'): This product is amazing, I love it!
Prediction: LABEL_1 | Confidence: 0.95

Enter a review (or 'exit'): Total waste of money.
Prediction: LABEL_0 | Confidence: 0.97
```

---

## 🌐 Run as an API (optional)

### 1. Start FastAPI server
```bash
uvicorn inference.api:app --reload
```

### 2. Test it
- Open browser: http://127.0.0.1:8000/docs
- Paste a sample review and get prediction

---

## 🧠 Model Details

| Model | Base | Parameters | Epochs | Accuracy |
|-------|------|------------|--------|----------|
| DistilBERT | distilbert-base-uncased | ~66M | 3 | ~90% (approx.) |

---

## 📌 Key Features

- ✅ Lightweight and fast transformer (DistilBERT)
- ✅ Binary sentiment classification
- ✅ Easy-to-run training pipeline
- ✅ Ready-to-use CLI and REST API
- ✅ Clean project structure and reproducible setup

---

## 👨‍💻 Author

**Ujunwa Njoku**  
LLM & DevOps Engineer

---

## ⭐️ Show Your Support

If you find this project helpful, feel free to ⭐️ the repository and share it!

---
