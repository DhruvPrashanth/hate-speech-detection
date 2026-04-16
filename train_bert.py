import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils import clean_text
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset.csv") #loads the dataset

df = df.sample(frac=1).reset_index(drop=True)
df = df.drop_duplicates()


print(df.head())
print(df.columns)
print(df['label'].unique())

from collections import Counter
print(Counter(df['label']))

texts = [clean_text(t) for t in df['text'].tolist()] #used to clean the dataset
labels = df['label'].tolist()

#slicing lists to train faster
texts = texts[:3000]
labels = labels[:3000]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, stratify=labels
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class HateDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateDataset(train_texts, train_labels)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

for epoch in range(4):
    print(f"Starting epoch {epoch}")

    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"Step {i}")

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if i % 50 == 0:
            print(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

model.save_pretrained("model_bert/")
tokenizer.save_pretrained("model_bert/")

model.eval()

val_dataset = HateDataset(val_texts, val_labels)
val_loader = DataLoader(val_dataset, batch_size=8)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(batch['labels'].tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Hate", "Offensive", "Neutral"],
            yticklabels=["Hate", "Offensive", "Neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BERT")
plt.show()