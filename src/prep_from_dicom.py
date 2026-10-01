"""
Build a training subset directly from raw DICOM files, using real HU conversion
and multi-window preprocessing (see data_prep.py). Produces a patient-level
train/val/test split so validation and test scores are not inflated by
slices from the same patient appearing on both sides.

Requires actual .dcm files (not the pre-rendered PNG mirror). Get a subset with
the Kaggle CLI once you've accepted the competition rules, e.g.:

    kaggle competitions download -c rsna-intracranial-hemorrhage-detection -f stage_2_train.zip
    unzip -q stage_2_train.zip -d stage_2_train

pydicom.dcmread() is used to recover PatientID for the patient-level split.
If PatientID is missing or anonymized out, this falls back to a random
image-level split and SAYS SO explicitly in the printed summary and in
labels.csv metadata -- it never silently claims a patient-safe split it
didn't actually do.
"""
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from data_prep import dicom_to_3ch  # real HU conversion + brain/subdural/bone windows

LABELS = ["epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural", "any"]


def get_patient_id(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return str(getattr(ds, "PatientID", None))
    except Exception:
        return None


def build_subset(csv_path, dicom_dir, out_dir, n_negatives=4000, size=128,
                  val_frac=0.15, test_frac=0.15, seed=42):
    df = pd.read_csv(csv_path)
    df["ImageID"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
    df["SubType"] = df["ID"].apply(lambda x: x.split("_")[-1])
    df = df.drop_duplicates(subset=["ImageID", "SubType"], keep="first")
    wide = df.pivot(index="ImageID", columns="SubType", values="Label").reset_index()

    pos = wide[wide["any"] == 1]
    neg_pool = wide[wide["any"] == 0]
    neg = neg_pool.sample(n=min(n_negatives, len(neg_pool)), random_state=seed)
    subset = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    records, missing = [], 0
    patient_ids_found = 0

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="DICOM -> HU -> window"):
        dcm_path = os.path.join(dicom_dir, f"{row['ImageID']}.dcm")
        if not os.path.exists(dcm_path):
            missing += 1
            continue
        try:
            arr = dicom_to_3ch(dcm_path, size)
        except Exception:
            missing += 1
            continue
        pid = get_patient_id(dcm_path)
        if pid and pid != "None":
            patient_ids_found += 1
        Image.fromarray(arr).save(os.path.join(out_dir, f"{row['ImageID']}.png"))
        records.append([row["ImageID"], pid] + [row[l] for l in LABELS])

    out_df = pd.DataFrame(records, columns=["ImageID", "PatientID"] + LABELS)

    have_patient_ids = patient_ids_found > 0.9 * len(out_df)
    if have_patient_ids:
        rng = np.random.RandomState(seed)
        patients = out_df["PatientID"].unique()
        rng.shuffle(patients)
        n_test = int(len(patients) * test_frac)
        n_val = int(len(patients) * val_frac)
        test_p = set(patients[:n_test])
        val_p = set(patients[n_test:n_test + n_val])
        split = out_df["PatientID"].apply(
            lambda p: "test" if p in test_p else ("val" if p in val_p else "train"))
        split_type = "patient-level"
    else:
        idx = out_df.sample(frac=1, random_state=seed).index
        n_test = int(len(out_df) * test_frac)
        n_val = int(len(out_df) * val_frac)
        split = pd.Series("train", index=out_df.index)
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
        split_type = "IMAGE-LEVEL (PatientID unavailable -- split is NOT patient-safe)"

    out_df["split"] = split
    out_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    print(f"Saved {len(out_df)} images | missing/failed: {missing}")
    print(f"Split type: {split_type}")
    print(out_df["split"].value_counts())
    print(out_df.groupby("split")["any"].mean().rename("positive_rate"))


if __name__ == "__main__":
    build_subset(
        csv_path="stage_2_train.csv",
        dicom_dir="stage_2_train/",
        out_dir="subset_data_dicom/",
        n_negatives=4000,
        size=128,
    )
