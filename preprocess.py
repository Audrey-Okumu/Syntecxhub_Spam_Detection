# preprocess.py
# Loads the dataset and cleans the text

import re
import pandas as pd

# A list of common English words that carry no meaning for spam detection
STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','a','an','the','and','but',
    'if','or','as','of','at','by','for','with','about','into','through',
    'to','from','up','in','out','on','off','then','here','there','when',
    'where','how','all','both','each','more','other','some','no','not',
    'only','same','so','than','too','very','can','will','just','now'
}

def load_data(filepath='data/SMSSpamCollection'):
    """Read the tab-separated file into a dataframe."""
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'text'])
    print(f"Loaded {len(df)} messages")
    print(df['label'].value_counts())
    return df

def clean_text(text):
    """Clean a single SMS message."""
    text = text.lower()                        # make everything lowercase
    text = re.sub(r'[^a-z\s]', '', text)      # remove punctuation & numbers
    tokens = text.split()                      # split into individual words
    tokens = [w for w in tokens              #remove stopwords & short words
              if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)                    # join back into a string

def preprocess(filepath='data/SMSSpamCollection'):
    """Full pipeline: load → clean → return features and labels."""
    df = load_data(filepath)
    print("\nCleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    X = df['clean_text']                       # features (the cleaned messages)
    y = (df['label'] == 'spam').astype(int)   # labels: 1 = spam, 0 = ham
    print("Done cleaning.")
    return X, y, df

if __name__ == '__main__':
    X, y, df = preprocess()
    print("\n--- Sample Output ---")
    for i in range(3):
        print(f"\nOriginal:  {df['text'].iloc[i]}")
        print(f"Cleaned:   {df['clean_text'].iloc[i]}")
        print(f"Label:     {df['label'].iloc[i]}")