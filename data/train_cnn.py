"""
character-level CNN 학습 + 평가 (PyTorch).
baseline과 동일한 stratified 70/15/15 split(random_state=42)을 재현해 공정 비교.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 200
EMB_DIM = 32
NUM_FILTERS = 64
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 데이터 불러오기 + baseline과 동일한 split 재현
df = pd.read_csv("features.csv")
y_all = (df["label"] == "phishing").astype(int).values
urls_all = df["url"].values
idx = np.arange(len(df))

idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx, y_all, test_size=0.30, stratify=y_all, random_state=42
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

urls_train, urls_val, urls_test = urls_all[idx_train], urls_all[idx_val], urls_all[idx_test]
print(f"train: {len(urls_train)}  val: {len(urls_val)}  test: {len(urls_test)}")


# 2) 문자 -> 숫자 vocab
class CharVocab:
    def __init__(self, urls):
        chars = sorted(set("".join(urls)))
        self.char2id = {c: i + 2 for i, c in enumerate(chars)}  # 0=PAD, 1=UNK
        self.pad_id = 0
        self.unk_id = 1
        self.size = len(self.char2id) + 2

    def encode(self, url: str, max_len: int = MAX_LEN) -> np.ndarray:
        ids = [self.char2id.get(c, self.unk_id) for c in url[:max_len]]
        ids += [self.pad_id] * (max_len - len(ids))
        return np.array(ids, dtype=np.int64)


vocab = CharVocab(urls_train)
print(f"vocab size: {vocab.size}")


# 3) Dataset
class UrlDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls = urls
        self.labels = labels

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, i):
        x = vocab.encode(self.urls[i])
        y = np.float32(self.labels[i])
        return torch.from_numpy(x), torch.tensor(y)


train_loader = DataLoader(UrlDataset(urls_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(UrlDataset(urls_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(UrlDataset(urls_test, y_test), batch_size=BATCH_SIZE)


# 4) 모델
class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, num_filters=NUM_FILTERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(emb_dim, num_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(emb_dim, num_filters, kernel_size=5)
        self.fc = nn.Linear(num_filters * 2, 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (batch, emb_dim, seq_len)
        c3 = F.relu(self.conv3(emb))
        c5 = F.relu(self.conv5(emb))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        p5 = F.max_pool1d(c5, c5.size(2)).squeeze(2)
        combined = torch.cat([p3, p5], dim=1)
        return self.fc(combined).squeeze(1)  # logits (sigmoid는 loss/평가 때 적용)


model = CharCNN(vocab.size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()  # sigmoid + BCE를 한 번에, 수치적으로 더 안정적


def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend((probs >= 0.5).astype(int))
            labels.extend(y.numpy().astype(int))
    return labels, preds


# 5) 학습 루프
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    val_labels, val_preds = evaluate(val_loader)
    acc = np.mean(np.array(val_labels) == np.array(val_preds))
    print(f"epoch {epoch}/{EPOCHS}  train_loss={total_loss / len(urls_train):.4f}  val_acc={acc:.4f}")

# 6) 최종 val 평가 (baseline과 동일 포맷)
val_labels, val_preds = evaluate(val_loader)
print("\n=== CharCNN (validation) ===")
print(classification_report(val_labels, val_preds, target_names=["benign", "phishing"], digits=4))
print("confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(val_labels, val_preds))
