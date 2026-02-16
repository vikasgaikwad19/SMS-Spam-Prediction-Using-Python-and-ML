import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')
# 2. Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})
# 3. Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data["message"] = data["message"].apply(preprocess)
# 4. Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["message"])
y = data["label"]
# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# 6. Train model
model = MultinomialNB()
model.fit(X_train, y_train)
# 7. Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# 8. Test on new SMS
sample_sms = ["Congratulations! You won a free lottery ticket"]
sample_sms = [preprocess(sms) for sms in sample_sms]
sample_vector = vectorizer.transform(sample_sms)

prediction = model.predict(sample_vector)
print("Spam" if prediction[0] == 1 else "Ham")
