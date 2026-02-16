# 📩 SMS Spam Detection Using Machine Learning

## 📌 Project Overview

This project implements an **SMS Spam Detection System** using **Machine Learning and Natural Language Processing (NLP)**.  
The system classifies SMS messages into:

- ✅ Ham (Not Spam)
- ⚠ Spam

It helps in filtering unwanted or fraudulent messages automatically.

---

## 🎯 Problem Statement

Spam messages are a major problem in communication systems.  
The goal of this project is to build a machine learning model that can automatically detect whether an SMS message is **spam** or **ham**.

---

## 🧠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Natural Language Processing (NLP)  
- TF-IDF Vectorization  

---


---

## 📊 Dataset

### Recommended Dataset
SMS Spam Collection Dataset:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 📝 Dataset Format

The dataset (`spam.csv`) should contain the following columns:

| label | message |
|------|---------|
| ham  | Hello, how are you? |
| spam | Congratulations! You won a prize |

---

## 🚀 How to Run the Project

### ✅ 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
## 2️⃣ Train the Model
python main.py
3️⃣ Predict a Message
python main.py
