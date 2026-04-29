# Converts text to numbers, trains two models, evaluates them, saves the best one

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess

def train_and_evaluate():

    # --- STEP A: Load and preprocess the data ---
    X, y, _ = preprocess()

    # --- STEP B: Split into training and test sets ---
    # 80% of data for training, 20% for testing
    # stratify=y ensures both sets have same ratio of spam/ham
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining messages: {len(X_train)}")
    print(f"Testing messages:  {len(X_test)}")

    # --- STEP C: Define two pipelines ---
    # A pipeline chains the vectorizer + model into one object
    # TfidfVectorizer converts text to numbers (word importance scores)
    # ngram_range=(1,2) means it looks at single words AND pairs of words
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf',   MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf',   LogisticRegression(max_iter=1000, random_state=42))
        ])
    }

    best_f1 = 0
    best_name = None
    best_pipeline = None

    # --- STEP D: Train and evaluate each model ---
    for name, pipeline in models.items():
        print(f"\n{'='*50}")
        print(f"  Model: {name}")
        print(f"{'='*50}")

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)

        # Print the full evaluation report
        print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

        # Print the confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Ham (correct):    {cm[0][0]}")
        print(f"  Ham called Spam (bad): {cm[0][1]}")
        print(f"  Spam called Ham (bad): {cm[1][0]}")
        print(f"  True Spam (correct):   {cm[1][1]}")

        # Track the best model by spam F1-score
        report = classification_report(y_test, y_pred,
                   target_names=['ham','spam'], output_dict=True)
        spam_f1 = report['spam']['f1-score']
        if spam_f1 > best_f1:
            best_f1 = spam_f1
            best_name = name
            best_pipeline = pipeline

    # --- STEP E: Save the best pipeline ---
    joblib.dump(best_pipeline, 'models/pipeline.pkl')
    print(f"\nBest model: {best_name}  (spam F1 = {best_f1:.4f})")
    print("Pipeline saved to models/pipeline.pkl")

if __name__ == '__main__':
    train_and_evaluate()