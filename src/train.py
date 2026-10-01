import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import ICHDataset, LABELS
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_weights(df):
    pos_counts = df[LABELS].sum().values
    total = len(df)
    return torch.tensor((total - pos_counts) / (pos_counts + 1e-6), dtype=torch.float32)


def train(epochs=8, batch_size=64, lr=1e-4, img_dir="subset_data_dicom",
          csv="subset_data_dicom/labels.csv"):
    """
    Expects labels.csv to have a 'split' column with values train/val/test,
    as written by prep_from_dicom.py. Checkpoint selection uses 'val' only;
    'test' stays untouched until evaluate.py is run, so the reported test
    score is a genuine held-out number, not one used for any decision.
    """
    full_df = pd.read_csv(csv)
    if "split" not in full_df.columns:
        raise ValueError(
            "labels.csv has no 'split' column. Run prep_from_dicom.py, or pass "
            "csv/img_dir pointing at a labels.csv that has one."
        )
    train_df = full_df[full_df["split"] == "train"]

    train_dl = DataLoader(ICHDataset(csv, img_dir, train=True, split="train"),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(ICHDataset(csv, img_dir, train=False, split="val"),
                         batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"train: {len(train_df)} | val: {(full_df['split']=='val').sum()} "
          f"| test (untouched during training): {(full_df['split']=='test').sum()}")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=get_class_weights(train_df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        t0, total_loss = time.time(), 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                out = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
                preds.append(out)
                gts.append(labels.numpy())
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        try:
            mean_auc = np.mean([roc_auc_score(gts[:, i], preds[:, i]) for i in range(6)])
        except ValueError:
            mean_auc = float("nan")

        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_dl):.4f} "
              f"| VAL_mean_AUC={mean_auc:.4f} | time={time.time()-t0:.1f}s")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"Best VALIDATION mean AUC: {best_auc:.4f} (saved to best_model.pt)")
    print("Run evaluate.py now to get the untouched TEST split score.")


if __name__ == "__main__":
    train()
