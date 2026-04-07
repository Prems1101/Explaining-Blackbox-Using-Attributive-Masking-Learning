from transformers import pipeline

def load_sentiment_model(model_path=None):
    if model_path:
        return pipeline("sentiment-analysis", model=model_path)
    else:
        return pipeline("sentiment-analysis")