import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path


# from dataset

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "mps"  # MPS for Mac M4
NUM_WORKERS = 4
N_CLASSES = 2  # healthy vs unhealthy
N_DOMAINS = 3  # shelled, unshelled, mixed



class TamarindDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.df = df
        self.transform = transform
        self.cls2idx = {"healthy":0, "unhealthy":1}
        self.dom2idx = {"shelled":0, "unshelled":1, "mixed":2}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # row = self.df.iloc[idx]

        row = self.df.iloc[idx]
        # print("DEBUG row['class'] =", row['class'])  # <-- add this
        img = Image.open(row['path']).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Fix: handle both string and one-hot
        raw_label = row['class']
        if isinstance(raw_label, str):
            label = self.cls2idx[raw_label]
        else:
            # assume one-hot or numeric
            arr = np.array(raw_label)
            if arr.ndim > 0 and arr.size > 1:
                label = int(arr.argmax())
            else:
                label = int(arr)

        domain = self.dom2idx.get(row['domain'], 0)
        return img, label, domain

# train_tfms = T.Compose([
#     T.RandomResizedCrop(IMG_SIZE),
#     T.RandomHorizontalFlip(),
#     T.ColorJitter(0.2,0.2,0.2,0.02),
#     T.ToTensor(),
#     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2,0.2,0.2,0.02),
    T.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


val_tfms = T.Compose([
    T.Resize((IMG_SIZE,IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


class FiLMAdapter(nn.Module):
    def __init__(self, feat_dim, n_domains):
        super().__init__()
        self.gamma = nn.Embedding(n_domains, feat_dim)
        self.beta = nn.Embedding(n_domains, feat_dim)

    def forward(self, feats, domain_idx):
        gamma = self.gamma(domain_idx)  # [B, feat_dim]
        beta = self.beta(domain_idx)    # [B, feat_dim]
        return feats * (1 + gamma) + beta  # [B, feat_dim]

# -----------------------------
# Teacher model: Swin-T + FiLM
# -----------------------------
class TeacherModel(nn.Module):
    def __init__(self, n_classes=2, n_domains=3):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        # print("DEBUG feats shape:", feats.shape)
        self.film = FiLMAdapter(feat_dim, n_domains)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x, domain_idx):
        feats = self.backbone(x)               # [B, 768]
        if feats.ndim == 1:                    # safeguard
            feats = feats.unsqueeze(0)
        # print("DEBUG feats shape:", feats.shape)
        feats = self.film(feats, domain_idx)
        logits = self.classifier(feats)        # [B, 2]
        return logits, feats

# -----------------------------
# Contrastive loss helper
# -----------------------------
def contrastive_loss(feats, labels, temp=0.1):
    # Normalize
    feats = F.normalize(feats, dim=1)
    logits = feats @ feats.T / temp
    targets = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    loss = F.cross_entropy(logits, targets.argmax(dim=1))
    return loss

# -----------------------------
# Training loop
# -----------------------------
def train_epoch(model, loader, opt, ce_loss, device):
    model.train()
    total_loss = 0
    for imgs, labels, domains in loader:
        imgs, labels, domains = imgs.to(device), labels.to(device), domains.to(device)
        # imgs, labels, domains = imgs.to(device), labels.to(device), domains.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        logits, feats = model(imgs, domains)
        loss_ce = ce_loss(logits, labels)

        # opt.zero_grad()
        # logits, feats = model(imgs, domains)
        # loss_ce = ce_loss(logits, labels)
        loss_con = contrastive_loss(feats, labels)
        loss = loss_ce + 0.1 * loss_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels, domains in loader:
        imgs, labels, domains = imgs.to(device), labels.to(device), domains.to(device)
        logits, _ = model(imgs, domains)
        preds = logits.argmax(1).cpu().numpy()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
    print(classification_report(y_true, y_pred, target_names=["healthy","unhealthy"]))
    return np.mean(np.array(y_true)==np.array(y_pred))

def main():
    train_ds = TamarindDataset("./output/split_train.csv", transform=train_tfms)
    val_ds = TamarindDataset("./output/split_val.csv", transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TeacherModel(N_CLASSES, N_DOMAINS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # ce_loss = nn.CrossEntropyLoss()
    class_weights = torch.tensor([1 / 6408, 1 / 2024], dtype=torch.float).to(DEVICE)
    class_weights = class_weights / class_weights.sum()  # normalize
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        loss = train_epoch(model, train_loader, opt, ce_loss, DEVICE)
        acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss {loss:.4f}, Val Acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "teacher_best.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training done. Best val acc:", best_acc)

    # for epoch in range(EPOCHS):
    #     print(f"Epoch {epoch+1}/{EPOCHS}")
    #     loss = train_epoch(model, train_loader, opt, ce_loss, DEVICE)
    #     acc = evaluate(model, val_loader, DEVICE)
    #     print(f"Epoch {epoch+1}/{EPOCHS}, Loss {loss:.4f}, Val Acc {acc:.3f}")
    #     if acc > best_acc:
    #         best_acc = acc
    #         torch.save(model.state_dict(), "teacher_best.pth")
    # print("Training done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()