# 01-LLM.py
# Basic Tokenization & Sentiment Analysis with Visualization

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# -------------------------------
# Load Pre-trained Model & Tokenizer
# -------------------------------
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# -------------------------------
# Example 1: Tokenization
# -------------------------------
text = "Large Language Models are transforming AI applications!"
tokens = tokenizer.tokenize(text)
print("🔹 Original Text:", text)
print("🔹 Tokens:", tokens)

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("🔹 Token IDs:", token_ids)

# -------------------------------
# Example 2: Sentiment Analysis
# -------------------------------
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

examples = [
    "I love working with machine learning models!",
    "This project is so complicated and frustrating.",
    "The weather is neutral today."
]

results = [sentiment_analyzer(text)[0] for text in examples]

print("\n📊 Sentiment Analysis Results:")
for text, result in zip(examples, results):
    print(f"Text: {text}")
    print(f" → Label: {result['label']}, Confidence: {result['score']:.4f}\n")

# -------------------------------
# Example 3: Visualization
# -------------------------------
labels = [res['label'] for res in results]
scores = [res['score'] for res in results]

plt.figure(figsize=(8, 5))
bars = plt.barh(examples, scores, color=["green" if lbl == "POSITIVE" else "red" for lbl in labels])
plt.xlabel("Confidence Score")
plt.title("Sentiment Analysis Results")

# Add labels on bars
for bar, label, score in zip(bars, labels, scores):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{label} ({score:.2f})", va='center')

plt.tight_layout()
plt.show()
