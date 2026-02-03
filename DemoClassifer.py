"""
Improved Multi-Class Text Classification System
Features:
- Cross-validation
- Hyperparameter tuning
- Enhanced preprocessing
- Better data balancing
- Model persistence
- Comprehensive evaluation
"""

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, precision_score, recall_score)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('movie_reviews', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import WordNetLemmatizer

print("="*60)
print("STEP 1: DATA LOADING")
print("="*60)

# Load News Data (subset for faster training)
news_data = fetch_20newsgroups(
    subset="train",
    categories=["rec.sport.baseball", "sci.space", "talk.politics.misc"],
    shuffle=True,
    random_state=42
)

news_df = pd.DataFrame({
    "text": news_data.data,
    "label": ["news"] * len(news_data.data)
})

print(f"✓ News samples loaded: {len(news_df)}")

# Load Spam Data
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
spam_df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
spam_df = spam_df[spam_df["label"] == "spam"].copy()
spam_df["label"] = "spam"

print(f"✓ Spam samples loaded: {len(spam_df)}")

# Load Sentiment Data (Movie Reviews)
sent_texts = []
for fileid in movie_reviews.fileids():
    sent_texts.append(" ".join(movie_reviews.words(fileid)))

sent_df = pd.DataFrame({
    "text": sent_texts,
    "label": ["sentiment"] * len(sent_texts)
})

print(f"✓ Sentiment samples loaded: {len(sent_df)}")

# Add extra sentiment samples
extra_sentiments = pd.DataFrame({
    "text": [
        "I am very happy today",
        "This is absolutely wonderful",
        "I feel sad and disappointed",
        "I love this so much",
        "I am extremely angry right now",
        "This made me feel amazing",
        "I am really upset",
        "I feel excited and thrilled",
        "This was a terrible experience",
        "I am feeling peaceful and calm"
    ],
    "label": ["sentiment"] * 10
})

# Combine all datasets
df = pd.concat([news_df, spam_df, sent_df, extra_sentiments], ignore_index=True)

print("\nClass Distribution (Before Balancing):")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")

# Balance the dataset by undersampling
print("\n" + "="*60)
print("STEP 2: DATA BALANCING")
print("="*60)

min_samples = df["label"].value_counts().min()
balanced_dfs = []

for label in df["label"].unique():
    label_df = df[df["label"] == label]
    sampled_df = label_df.sample(n=min_samples, random_state=42)
    balanced_dfs.append(sampled_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class Distribution (After Balancing):")
print(df_balanced["label"].value_counts())
print(f"Total samples: {len(df_balanced)}")

# Enhanced Text Preprocessing
print("\n" + "="*60)
print("STEP 3: TEXT PREPROCESSING")
print("="*60)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Enhanced text cleaning with lemmatization"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Handle contractions
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(words)

print("Cleaning text data...")
df_balanced["clean_text"] = df_balanced["text"].apply(clean_text)

# Remove empty texts after cleaning
df_balanced = df_balanced[df_balanced["clean_text"].str.len() > 0].reset_index(drop=True)

print(f"✓ Text cleaning complete. Final dataset size: {len(df_balanced)}")

# Prepare features and labels
X = df_balanced["clean_text"]
y = df_balanced["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Enhanced TF-IDF Vectorization
print("\n" + "="*60)
print("STEP 4: FEATURE EXTRACTION (TF-IDF)")
print("="*60)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,            # Ignore terms appearing in fewer than 2 documents
    max_df=0.8,          # Ignore terms appearing in more than 80% of documents
    sublinear_tf=True    # Apply sublinear tf scaling
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Feature matrix shape: {X_train_tfidf.shape}")
print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")

# Model Training with Cross-Validation
print("\n" + "="*60)
print("STEP 5: MODEL TRAINING & CROSS-VALIDATION")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
cv_results = {}

for name, model in models.items():
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores
    
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full training set
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")

# Display model comparison
print("\n" + "="*60)
print("STEP 6: MODEL COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='accuracy', ascending=False)
print("\n" + results_df.to_string())

# Hyperparameter Tuning for Best Model
print("\n" + "="*60)
print("STEP 7: HYPERPARAMETER TUNING")
print("="*60)

best_model_name = results_df.index[0]
print(f"\nTuning hyperparameters for: {best_model_name}")

if best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    
elif best_model_name == "Naive Bayes":
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }
    base_model = MultinomialNB()
    
elif best_model_name == "SVM":
    param_grid = {
        'C': [0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge']
    }
    base_model = LinearSVC(max_iter=2000, random_state=42)
    
else:  # Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42)

print("Performing Grid Search...")
grid_search = GridSearchCV(
    base_model, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_tfidf, y_train)

print(f"\n✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best CV score: {grid_search.best_score_:.4f}")

# Use the best model
best_model = grid_search.best_estimator_

# Final Evaluation
print("\n" + "="*60)
print("STEP 8: FINAL MODEL EVALUATION")
print("="*60)

y_pred = best_model.predict(X_test_tfidf)

print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
labels = sorted(df_balanced["label"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title(f"Confusion Matrix - {best_model_name} (Tuned)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/HP/Desktop/WeIntern/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")

# Cross-validation comparison plot
plt.figure(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot()
plt.title("Cross-Validation Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/HP/Desktop/WeIntern/cv_comparison.png', dpi=300, bbox_inches='tight')
print("✓ CV comparison plot saved as 'cv_comparison.png'")

# Save the model and vectorizer
print("\n" + "="*60)
print("STEP 9: MODEL PERSISTENCE")
print("="*60)

model_data = {
    'model': best_model,
    'vectorizer': vectorizer,
    'model_name': best_model_name,
    'labels': labels,
    'best_params': grid_search.best_params_,
    'test_accuracy': accuracy_score(y_test, y_pred)
}

with open('C:/Users/HP/Desktop/WeIntern/text_classifier_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved as 'text_classifier_model.pkl'")

# Prediction Function
def predict_category(text, model_data):
    """Predict the category of new text"""
    try:
        # Clean the text
        text_clean = clean_text(text)
        
        if len(text_clean) == 0:
            return "Error: Text is empty after cleaning"
        
        # Vectorize
        text_tfidf = model_data['vectorizer'].transform([text_clean])
        
        # Predict
        prediction = model_data['model'].predict(text_tfidf)[0]
        
        # Get prediction probabilities if available
        if hasattr(model_data['model'], 'predict_proba'):
            proba = model_data['model'].predict_proba(text_tfidf)[0]
            confidence = max(proba)
            return f"{prediction} (confidence: {confidence:.2%})"
        elif hasattr(model_data['model'], 'decision_function'):
            decision = model_data['model'].decision_function(text_tfidf)[0]
            return f"{prediction} (decision score: {max(decision):.2f})"
        else:
            return prediction
            
    except Exception as e:
        return f"Error: {str(e)}"

# Test predictions
print("\n" + "="*60)
print("STEP 10: EXAMPLE PREDICTIONS")
print("="*60)

test_texts = [
    "NASA launched a satellite into orbit around Mars",
    "Win free iPhone now click here limited time offer",
    "I feel very happy today, this is wonderful",
    "The baseball team won the championship game yesterday",
    "Claim your prize now! You've been selected as a winner!",
    "This movie was absolutely terrible, I hated every minute"
]

print("\nPredicting categories for sample texts:\n")
for text in test_texts:
    prediction = predict_category(text, model_data)
    print(f"Text: {text[:60]}...")
    print(f"→ Prediction: {prediction}\n")

print("="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✓ Model saved to: text_classifier_model.pkl")
print(f"✓ Confusion matrix saved to: confusion_matrix.png")
print(f"✓ CV comparison saved to: cv_comparison.png")
print("\n" + "="*60)