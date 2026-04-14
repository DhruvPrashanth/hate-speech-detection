import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils import clean_text

df = pd.read_csv("dataset.csv") #loads the dataset

texts = [clean_text(t) for t in df['text'].tolist()] #used to clean the dataset
labels = df['label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class HateDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateDataset(train_texts, train_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

for epoch in range(4):
    print(f"Starting epoch {epoch}")

    for i, batch in enumerate(loader):
        if i % 50 == 0:
            print(f"Step {i}")   # 👈 ADD THIS

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

# Save model
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")