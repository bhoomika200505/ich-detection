"""Build the training subset from the pre-processed RSNA PNG dataset
(guiferviz/rsna_stage1_png_128): all positives + n_negatives negatives."""
import os
import shutil
import pandas as pd

LABELS = ["epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural", "any"]


def build_subset_from_png(csv_path, img_dir, out_dir, n_negatives=4000):
    df = pd.read_csv(csv_path)
    df["ImageID"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
    df["SubType"] = df["ID"].apply(lambda x: x.split("_")[-1])
    df = df.drop_duplicates(subset=["ImageID", "SubType"], keep="first")
    wide = df.pivot(index="ImageID", columns="SubType", values="Label").reset_index()

    pos = wide[wide["any"] == 1]
    neg_pool = wide[wide["any"] == 0]
    neg = neg_pool.sample(n=min(n_negatives, len(neg_pool)), random_state=42)
    subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    records, missing = [], 0
    for _, row in subset.iterrows():
        src = os.path.join(img_dir, f"{row['ImageID']}.png")
        if not os.path.exists(src):
            missing += 1
            continue
        shutil.copy(src, os.path.join(out_dir, f"{row['ImageID']}.png"))
        records.append([row["ImageID"]] + [row[l] for l in LABELS])

    out_df = pd.DataFrame(records, columns=["ImageID"] + LABELS)
    out_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    print(f"Saved {len(out_df)} images | positive: {int(out_df['any'].sum())} "
          f"| negative: {int((out_df['any'] == 0).sum())} | missing: {missing}")


if __name__ == "__main__":
    build_subset_from_png(
        csv_path="rsna_png_data/stage_1_train.csv",
        img_dir="rsna_png_data/stage_1_train_images",
        out_dir="subset_data",
        n_negatives=4000,
    )
