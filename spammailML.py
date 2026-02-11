import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# --- 1. DATA PREPARATION / VERİ HAZIRLIĞI ---
# Create a sample dataset for spam detection
# Spam tespiti için örnek bir veri seti oluşturuyoruz
data = {
    'text': [
        'Win a brand new car now!', 'Hello, how are you today?', 
        'Free entry to the contest', 'Meeting scheduled for tomorrow',
        'Urgent: claim your prize!', 'Can we grab coffee later?',
        'Congratulations, you won $1000', 'Don’t forget the report',
        'Call this number to win cash', 'Thanks for the help'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1: Spam, 0: Ham/Normal
}
df = pd.DataFrame(data)

# --- 2. VECTORIZATION / SAYISALLAŞTIRMA ---
# Convert text to numerical data using TF-IDF
# TF-IDF kullanarak metin verilerini sayısal verilere dönüştürüyoruz
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split data into training and test sets
# Veriyi eğitim ve test setlerine ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MODELING / MODELLEME ---
# Building a Logistic Regression model
# Lojistik Regresyon modeli oluşturuyoruz
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# --- 4. EVALUATION & VISUALIZATION / DEĞERLENDİRME VE GÖRSELLEŞTİRME ---
# Print the success metrics
# Başarı metriklerini yazdırıyoruz
print("Accuracy Score:", accuracy_score(y_test, log_pred))
print("\nClassification Report:\n", classification_report(y_test, log_pred))

# Generate ROC Curve
# ROC Eğrisi oluşturma
log_probs = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, log_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate / Yanlış Pozitif Oranı')
plt.ylabel('True Positive Rate / Doğru Pozitif Oranı')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Save the plot for GitHub / GitHub için grafiği kaydet
plt.savefig('roc_curve.png')
print("\nSuccess: 'roc_curve.png' has been saved!")
plt.show()
