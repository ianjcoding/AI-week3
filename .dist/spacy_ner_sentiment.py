# spacy_ner_sentiment.py (or notebook cell)
"""
Task: Use spaCy to extract product names/brands and perform a simple rule-based sentiment analysis.
"""

import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
from collections import Counter

# Load spaCy model (install first: python -m pip install -U spacy && python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Add EntityRuler to recognize brand/product patterns (expand with real brand/product examples)
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "BRAND", "pattern": [{"LOWER":"apple"}]},
    {"label": "BRAND", "pattern": [{"LOWER":"samsung"}]},
    {"label": "PRODUCT", "pattern": [{"LOWER":"echo"}, {"LOWER":"dot"}]},
    {"label": "PRODUCT", "pattern": [{"LOWER":"kindle"}]},
    # You can add many patterns or load from a file of known brands/products
]
ruler.add_patterns(patterns)

# Example reviews
reviews = [
    "I love my new Apple iPhone, battery life is excellent!",
    "The Samsung Galaxy camera is amazing but the battery is poor.",
    "Terrible quality — the headphones stopped working after two days.",
    "I enjoy the Kindle; reading is so smooth and the battery lasts long."
]

# Simple sentiment lexicons (expand in real project)
positive_words = {"love", "excellent", "amazing", "smooth", "good", "great", "enjoy", "best"}
negative_words = {"terrible", "poor", "bad", "stopped", "disappointed", "worst"}

results = []
for text in reviews:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Rule-based sentiment: count positive - negative tokens
    tokens = [tok.lemma_.lower() for tok in doc if not tok.is_stop and tok.is_alpha]
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    sentiment = "neutral"
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    results.append({
        "text": text,
        "entities": entities,
        "tokens": tokens,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "sentiment": sentiment
    })

# Print results
for r in results:
    print("Review:", r["text"])
    print("Entities:", r["entities"])
    print("Sentiment:", r["sentiment"], f"(+{r['pos_count']}, -{r['neg_count']})")
    print("-"*60)
