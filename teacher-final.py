import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_SPLIT_DIR = Path("./output")
# BATCH_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 20
LR = 2e-4
NUM_CLASSES = 2
DOMAINS = ["shelled", "unshelled", "mixed"]
DOMAIN2ID = {d: i for i, d in enumerate(DOMAINS)}
IMG_SIZE = 384
# IMG_SIZE = 224
GRID = (3, 3)  # 3x3 patches per image for MIL
TEMPERATURE_CL = 0.07
USE_DOMAIN_CONTRASTIVE = True
BACKBONE_INPUT = 224        # swin_tiny expects 224x224
SEED = 42
NUM_OF_WORKERS = 2


def set_seed(seed=SEED):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MPS (Apple Silicon) specific settings
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        # MPS doesn't have benchmark mode, but we can set other deterministic flags
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Try to enable torch deterministic algorithms (if available)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Older PyTorch versions
        try:
            torch.set_deterministic(True)
        except AttributeError:
            pass

    # Force NumPy to use single thread for reproducibility
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


# Initialize seeding at the very beginning
set_seed(SEED)

# Create a generator for DataLoader reproducibility
def get_dataloader_generator(seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    """
    Worker_init_fn for DataLoader to ensure each worker has different but deterministic seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)

class TamarindPatches(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        self.tfm_train = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            # Affine with only broadly-supported args
            A.Affine(translate_percent=(0.0, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            ToTensorV2(),
        ])
        self.tfm_eval = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def _grid_crops(self, img_tensor):
        C, H, W = img_tensor.shape
        # If padding overshot, center-crop back to IMG_SIZE
        if H != IMG_SIZE or W != IMG_SIZE:
            top = max(0, (H - IMG_SIZE) // 2)
            left = max(0, (W - IMG_SIZE) // 2)
            img_tensor = img_tensor[:, top:top + IMG_SIZE, left:left + IMG_SIZE]
            C, H, W = img_tensor.shape

        gh, gw = GRID
        ph, pw = H // gh, W // gw  # expect 128x128
        patches = []
        for i in range(gh):
            for j in range(gw):
                patches.append(img_tensor[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw])
        patches = torch.stack(patches, dim=0)  # [P, C, ph, pw]
        return patches

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = r["path"]
        # y = 0 if r["label"].lower() == "healthy" else 1
        y = 0 if r["class"].lower() == "healthy" else 1
        # d = DOMAIN2ID.get(r["domain"], 0)
        d = DOMAIN2ID.get(str(r.get("domain", "")).lower().strip(), 0)

        with Image.open(p) as im:
            im = im.convert("RGB")
            arr = np.array(im)

        tfm = self.tfm_train if self.augment else self.tfm_eval
        out = tfm(image=arr)["image"]  # tensor C,H,W
        patches = self._grid_crops(out)
        return patches, torch.tensor(y).long(), torch.tensor(d).long(), p


class FiLM(nn.Module):
    def __init__(self, feat_dim, num_domains=3):
        super().__init__()
        self.emb = nn.Embedding(num_domains, 128)
        self.to_gamma = nn.Linear(128, feat_dim)
        self.to_beta = nn.Linear(128, feat_dim)

    def forward(self, x, d):
        e = self.emb(d)                        # [B,128]
        gamma = self.to_gamma(e).unsqueeze(1)  # [B,1,D]
        beta = self.to_beta(e).unsqueeze(1)    # [B,1,D]
        return x * (1 + gamma) + beta          # [B,P,D]


class AttnMILPool(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, feats):
        a = self.attn(feats)                   # [B,P,1]
        w = torch.softmax(a, dim=1)            # [B,P,1]
        bag = (w * feats).sum(dim=1)           # [B,D]
        return bag, w

class TeacherNet(nn.Module):
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224", num_domains=3, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")

        self.feat_dim = getattr(self.backbone, "num_features", None)
        if not self.feat_dim:
            with torch.no_grad():
                dummy = torch.zeros(1,3,BACKBONE_INPUT,BACKBONE_INPUT)
                feat = self.backbone(dummy)
                self.feat_dim = feat.shape[-1]

        self.film = FiLM(self.feat_dim, num_domains=num_domains)
        self.mil = AttnMILPool(self.feat_dim)
        self.head = nn.Linear(self.feat_dim, num_classes)

    @torch.no_grad()
    def encode_patch(self, x):  # x: [B*P, C, H, W]
        return self.backbone(x)

    def forward(self, patches, domain_ids):
        # patches: [B, P, C, h, w] (h,w ~ 128)
        B, P, C, h, w = patches.shape

        # ✅ ensure float32 before interpolate (fixes NotImplementedError for 'Byte')
        x = patches.view(B*P, C, h, w).float()

        # Resize each patch to backbone's input
        if h != BACKBONE_INPUT or w != BACKBONE_INPUT:
            x = F.interpolate(x, size=(BACKBONE_INPUT, BACKBONE_INPUT),
                              mode="bilinear", align_corners=False)

        patch_feats = self.backbone(x)            # [B*P, D]
        patch_feats = patch_feats.view(B, P, -1)  # [B,P,D]
        film_feats  = self.film(patch_feats, domain_ids)
        bag, attn_w = self.mil(film_feats)        # [B,D], [B,P,1]
        logits = self.head(bag)                   # [B,2]
        return logits, bag, film_feats, attn_w


def supervised_contrastive_loss(bag_feats, domain_ids, temp=TEMPERATURE_CL):
    # z = nn.functional.normalize(bag_feats, dim=1)
    # sim = torch.matmul(z, z.T) / temp
    # mask = (domain_ids.unsqueeze(1) == domain_ids.unsqueeze(0)).float()
    # logits = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    # logp = logits.log_softmax(dim=1)
    # pos = logp * mask
    # denom = mask.sum(dim=1).clamp_min(1.0)
    # loss = -(pos.sum(dim=1) / denom).mean()
    # return loss

    z = nn.functional.normalize(bag_feats, dim=1)
    sim = torch.matmul(z, z.T) / temp  # [B,B]
    logits = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    logp = logits.log_softmax(dim=1)
    mask = (domain_ids.unsqueeze(1) == domain_ids.unsqueeze(0)).float()
    pos = logp * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    loss = -(pos.sum(dim=1) / denom).mean()
    return loss


def run_train(train_csv, val_csv, save_path=f"./saved-models"):
    model_save_path = Path(save_path)
    model_save_path.mkdir(exist_ok=True)
    model_file_path = model_save_path / f"{os.path.basename(__file__).split(".")[0]}_teacher_swin_film_mil.pt"

    train_ds = TamarindPatches(train_csv, augment=True)
    val_ds = TamarindPatches(val_csv, augment=False)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS, pin_memory=True, drop_last=True)
    # val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKERS, pin_memory=True)

    train_gen = get_dataloader_generator(SEED)
    val_gen = get_dataloader_generator(SEED + 7)

    pin = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=pin,  # Disable pin_memory for MPS
        worker_init_fn=seed_worker,
        generator=train_gen,
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=pin,
        worker_init_fn=seed_worker,
        generator=val_gen,
        drop_last=False
    )

    model = TeacherNet(num_domains=len(DOMAINS), num_classes=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ---- FIX: dynamic class weights + label smoothing
    train_counts = train_ds.df['class'].str.lower().value_counts()
    h_count = int(train_counts.get('healthy', 1))
    u_count = int(train_counts.get('unhealthy', 1))
    w_healthy = 1.0 / h_count
    w_unhealthy = 1.0 / u_count
    class_weights = torch.tensor([w_healthy, w_unhealthy], dtype=torch.float).to(DEVICE)
    class_weights = class_weights / class_weights.sum()
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_f1 = 0.0
    patience = 5
    counter = 0

    for epoch in range(1, EPOCHS+1):
        print(f"[Epoch {epoch}] >>> ", end="")
        model.train()
        tr_loss = 0.0
        for patches, y, d, _ in train_dl:
            patches, y, d = patches.to(DEVICE), y.to(DEVICE), d.to(DEVICE)

            logits, bag, _, _ = model(patches, d)
            loss = ce(logits, y)
            if USE_DOMAIN_CONTRASTIVE:
                loss += 0.05 * supervised_contrastive_loss(bag, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # ---- FIX: gradient clipping
            opt.step()
            tr_loss += loss.item() * patches.size(0)

        # validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for patches, y, d, _ in val_dl:
                patches, y, d = patches.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                logits, _, _, _ = model(patches, d)
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(y.cpu().tolist())

        acc = accuracy_score(gts, preds)
        mf1 = f1_score(gts, preds, average="macro")
        print(f" train_loss={tr_loss/len(train_ds):.4f}  val_acc={acc:.4f}  val_macroF1={mf1:.4f}")

        if mf1 > best_f1:
            best_f1 = mf1
            torch.save(model.state_dict(), str(model_file_path))
            print(f"  -> saved {str(model_file_path)} (best macro-F1 {best_f1:.4f})")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training done. Best macro-F1:", best_f1)
    return save_path


if __name__ == "__main__":
    best = run_train(DATA_SPLIT_DIR/"split_train.csv", DATA_SPLIT_DIR/"split_val_clean.csv")
    print("Best model saved at:", best)