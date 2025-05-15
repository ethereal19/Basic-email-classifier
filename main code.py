import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords
import re


data = {
    'email': [
        'Hey, want to hang out tomorrow?', 
        'Your Amazon order has shipped!',   
        'Meeting rescheduled to 10am.',     
        'Congratulations! You won a prize!',
        'Your bank statement is ready.'     
    ],
    'label': ['Personal', 'Promotions', 'Work', 'Spam', 'Finance']
}

df = pd.DataFrame(data)

# Preprocessing function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\W+', ' ', text)  
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

df['cleaned'] = df['email'].apply(clean_text)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))


new_email = "Here’s your monthly invoice from Dropbox"
cleaned_email = clean_text(new_email)
vectorized_email = vectorizer.transform([cleaned_email])
prediction = model.predict(vectorized_email)

print(f"\n📧 Email: {new_email}")
print(f"📁 Predicted Category: {prediction[0]}")
