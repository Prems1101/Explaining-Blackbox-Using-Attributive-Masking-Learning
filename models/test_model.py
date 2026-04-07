from load_model import load_sentiment_model

model = load_sentiment_model()

text = "This movie was absolutely amazing"
result = model(text)

print(result)