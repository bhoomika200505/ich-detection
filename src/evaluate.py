"""
Reports metrics on val AND test separately, so it's always clear which
number was used for checkpoint selection (val) and which is the untouched
held-out score (test). Requires labels.csv to have a 'split' column
(train/val/test), as written by prep_from_dicom.py.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from dataset import ICHDataset, LABELS
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_split(model, csv, img_dir, split_name):
    dl = DataLoader(ICHDataset(csv, img_dir, train=False, split=split_name),
                     batch_size=128, num_workers=2)
    preds, gts = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            gts.append(y.numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)

    print(f"\n=== {split_name.upper()} (n={len(gts)}) ===")
    for i, l in enumerate(LABELS):
        print(f"{l:18s} AUC {roc_auc_score(gts[:, i], preds[:, i]):.3f}")
    mean5 = np.mean([roc_auc_score(gts[:, i], preds[:, i]) for i in range(5)])
    print(f"Mean AUC (5 subtypes): {mean5:.4f}")
    print(f"Binary 'any' AUC: {roc_auc_score(gts[:, 5], preds[:, 5]):.4f}")
    pred_any = preds[:, 5] >= 0.5
    print(f"Recall: {recall_score(gts[:, 5], pred_any):.3f} "
          f"| Precision: {precision_score(gts[:, 5], pred_any):.3f}")
    print(f"Fraction positive: {gts[:, 5].mean():.3f}")
    return mean5


if __name__ == "__main__":
    csv = "subset_data_dicom/labels.csv"
    img_dir = "subset_data_dicom"

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device).eval()

    df = pd.read_csv(csv)
    if "split" not in df.columns or "test" not in df["split"].unique():
        raise ValueError(
            "No 'test' split found in labels.csv. This script needs the "
            "3-way split from prep_from_dicom.py -- it will not report a "
            "number labeled TEST on data that was used for anything else."
        )

    evaluate_split(model, csv, img_dir, "val")
    evaluate_split(model, csv, img_dir, "test")
