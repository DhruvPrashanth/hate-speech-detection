import matplotlib.pyplot as plt

models = ["DistilBERT", "BERT"]
accuracy = [1.0, 1.0]  # your results

plt.figure()
plt.bar(models, accuracy)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.ylim(0, 1.1)
plt.show()