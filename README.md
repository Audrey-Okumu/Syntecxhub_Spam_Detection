# 📧 Syntecxhub Spam Detection

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) and scikit-learn.

Built using the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 5,572 real labeled SMS messages.

---

## 📁 Project Structure

```
Syntecxhub_Spam_Detection/
├── data/
│   └── SMSSpamCollection       # Raw labeled dataset (tab-separated)
├── models/
│   └── pipeline.pkl            # Saved vectorizer + model (generated after training)
├── venv/                       # Virtual environment (not committed to git)
├── preprocess.py               # Loads and cleans text data
├── train.py                    # Trains models and saves the best pipeline
├── predict.py                  # Loads pipeline and classifies new messages
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas | Loading and exploring the dataset |
| scikit-learn | TF-IDF vectorization, model training, evaluation |
| joblib | Saving and loading the trained pipeline |

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Audrey-Okumu/Syntecxhub_Spam_Detection.git
cd Syntecxhub_Spam_Detection
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place the `SMSSpamCollection` file inside the `data/` folder:
```
data/SMSSpamCollection
```

---

## 🚀 How to Run

### Step 1 — Preprocess the data
```bash
python preprocess.py
```
Loads the dataset, cleans the text (lowercase, removes punctuation, removes stopwords), and prints sample output.

### Step 2 — Train the models
```bash
python train.py
```
Trains both **Naive Bayes** and **Logistic Regression**, prints precision/recall/F1 scores for each, and saves the best model to `models/pipeline.pkl`.

### Step 3 — Make predictions
```bash
python predict.py
```
Loads the saved pipeline and classifies a set of test messages. Edit the `test_messages` list in `predict.py` to try your own messages.

---

## 📊 How It Works

```
Raw SMS Text
     │
     ▼
preprocess.py   →   Lowercase + remove punctuation + remove stopwords
     │
     ▼
train.py        →   TF-IDF Vectorizer (text → numbers)
     │
     ▼
                →   Train Naive Bayes + Logistic Regression
     │
     ▼
                →   Evaluate with precision / recall / F1
     │
     ▼
models/pipeline.pkl  →  Saved for reuse
     │
     ▼
predict.py      →   Classify new messages as SPAM or HAM
```

---

## 📈 Model Results

Evaluated on 20% held-out test data (1,115 messages):

| Model | Spam Precision | Spam Recall | Spam F1 |
|---|---|---|---|
| Naive Bayes | 100% | 77% | 0.87 |
| Logistic Regression | 100% | 77% | **0.87** ✅ |

**Logistic Regression** was selected as the final model.

- **Precision 100%** — zero legitimate messages were wrongly flagged as spam
- **Recall 77%** — the model caught 77% of all actual spam messages
- **F1 Score 0.87** — strong balance between precision and recall

---

## 🧠 Key Concepts

**TF-IDF (Term Frequency–Inverse Document Frequency)** — converts text into numbers by scoring how important each word is in a message relative to the whole dataset. Words like "FREE" and "WIN" score high in spam messages.

**Naive Bayes** — a fast probabilistic classifier that works well with word frequency data.

**Logistic Regression** — a linear classifier that learns decision boundaries between spam and ham.

**Pipeline** — bundles the vectorizer and model into one object so both are saved and loaded together, making predictions on new data simple.

---

## 👤 Author

**Audrey Okumu**  
Built as part of the Syntecxhub ML projects.  
[GitHub](https://github.com/Audrey-Okumu) · [LinkedIn](https://www.linkedin.com/in/audrey-okumu-943221366/)

