"""
data_prep.py
Builds a small, class-balanced subset of the RSNA Intracranial Hemorrhage
dataset that trains fast on modest hardware (laptop CPU or a free Colab GPU).

Key compute-saving decisions:
- Only a few thousand images instead of ~750k in the full dataset.
- Keep ALL positive (hemorrhage) slices — epidural is <1% of the data, so we
  can't afford to randomly sample it away — plus a matched sample of negatives.
- Convert each DICOM to a small 3-channel PNG (brain/subdural/bone windows)
  ONCE, so training reads cheap PNGs instead of re-decoding DICOMs every epoch.
"""
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

LABELS = ["epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural", "any"]


def window_image(img, center, width):
    lo, hi = center - width / 2, center + width / 2
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return (img * 255).astype(np.uint8)


def dicom_to_3ch(path, size=128):
    """Brain / subdural / bone windows stacked as RGB channels — a standard,
    cheap way to give a single 2D slice 3 clinically-relevant 'views'."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    brain = window_image(img, 40, 80)
    subdural = window_image(img, 80, 200)
    bone = window_image(img, 600, 2800)
    stacked = np.stack([brain, subdural, bone], axis=-1)
    return np.array(Image.fromarray(stacked).resize((size, size)))


def build_subset(csv_path, dicom_dir, out_dir, n_negatives=4000, size=128):
    """
    csv_path: RSNA stage_2_train.csv (long format: ID_xxx_label, Label)
    dicom_dir: folder of .dcm files
    out_dir: where PNGs + labels.csv get written
    n_negatives: how many non-hemorrhage slices to keep (all positives kept)
    """
    df = pd.read_csv(csv_path)
    df["ImageID"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
    df["SubType"] = df["ID"].apply(lambda x: x.split("_")[-1])
    wide = df.pivot(index="ImageID", columns="SubType", values="Label").reset_index()

    pos = wide[wide["any"] == 1]
    neg_pool = wide[wide["any"] == 0]
    neg = neg_pool.sample(n=min(n_negatives, len(neg_pool)), random_state=42)
    subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    records = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Converting DICOMs"):
        dcm_path = os.path.join(dicom_dir, f"{row['ImageID']}.dcm")
        if not os.path.exists(dcm_path):
            continue
        try:
            arr = dicom_to_3ch(dcm_path, size)
        except Exception:
            continue
        Image.fromarray(arr).save(os.path.join(out_dir, f"{row['ImageID']}.png"))
        records.append([row["ImageID"]] + [row[l] for l in LABELS])

    out_df = pd.DataFrame(records, columns=["ImageID"] + LABELS)
    out_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print(f"Saved {len(out_df)} images | positive: {int(out_df['any'].sum())} | "
          f"negative: {int((out_df['any'] == 0).sum())}")
    print(out_df[LABELS].sum())


if __name__ == "__main__":
    build_subset(
        csv_path="stage_2_train.csv",
        dicom_dir="stage_2_train/",
        out_dir="subset_data/",
        n_negatives=4000,   # ~4-5k total images -> trains in well under an hour
        size=128,           # 128x128 is enough signal for hemorrhage vs. not
    )
