from explain.aml_explainer import AMLExplainer

explainer = AMLExplainer()

text = "This movie was absolutely amazing"

result = explainer.explain(text)

print(result["prediction"])

for word, score in result["word_importance"]:
    print(word, round(score, 3))