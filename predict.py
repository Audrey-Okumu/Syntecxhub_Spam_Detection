# Loads the saved pipeline and classifies new messages

import joblib

def load_pipeline(path='models/pipeline.pkl'):
    """Load the saved vectorizer + model from disk."""
    pipeline = joblib.load(path)
    print("Pipeline loaded successfully.")
    return pipeline

def predict_messages(messages, pipeline=None):
    """
    Takes a list of raw SMS strings.
    Returns a list of 'spam' or 'ham' labels.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    predictions = pipeline.predict(messages)
    labels = ['SPAM' if p == 1 else 'HAM' for p in predictions]
    return labels

if __name__ == '__main__':
    pipeline = load_pipeline()

    # Test messages 
    test_messages = [
        "Congratulations! You've won a FREE £1000 prize. Call now to claim!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your mobile account is suspended. Verify now at this link.",
        "Can you send me the notes from today's class?",
        "Win cash prizes! Text WIN to 80080. 150p per msg.",
        "I'll be home late tonight, don't wait up.",
    ]

    print(f"\n{'Message':<55} {'Result':>6}")
    print('-' * 63)

    results = predict_messages(test_messages, pipeline)
    for msg, label in zip(test_messages, results):
        short = msg[:52] + '...' if len(msg) > 52 else msg
        print(f"{short:<55} {label:>6}")