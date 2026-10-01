"""
Gradio demo. Accepts either:
  - a pre-processed 128x128 PNG/JPG (what the model was trained on), or
  - a raw DICOM (.dcm) file, converted live via the real HU + multi-window
    pipeline in data_prep.py.

Note on the DICOM path: the model's training data came from a pre-rendered
PNG mirror whose exact windowing parameters aren't published, so a live
DICOM run may not perfectly match the training distribution. This is
disclosed in the UI, not hidden.
"""
import os
import tempfile
from PIL import Image
import gradio as gr

from inference import predict
from data_prep import dicom_to_3ch

DICOM_DISCLAIMER = (
    "Research demo only, not for medical use. DICOM inputs are converted live "
    "via HU rescaling + brain/subdural/bone windowing (data_prep.py). The "
    "training data came from a pre-rendered PNG source whose exact windowing "
    "isn't published, so DICOM predictions may differ slightly from the "
    "reported validation/test metrics."
)


def run_inference(file):
    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        arr = dicom_to_3ch(path, size=128)
        tmp_path = tempfile.mktemp(suffix=".png")
        Image.fromarray(arr).save(tmp_path)
        preview = Image.fromarray(arr)
    else:
        tmp_path = path
        preview = Image.open(path).convert("RGB")

    probs, flagged = predict(tmp_path)
    if ext == ".dcm":
        os.remove(tmp_path)

    result_text = "\n".join(f"{k}: {v:.3f}" for k, v in probs.items())
    verdict = "Flagged: " + ", ".join(flagged) if flagged else "No hemorrhage detected"
    return preview, verdict, result_text


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.File(label="Upload a CT slice (.png/.jpg) or a raw DICOM (.dcm)"),
    outputs=[
        gr.Image(label="Processed slice"),
        gr.Textbox(label="Verdict"),
        gr.Textbox(label="Per-class probabilities"),
    ],
    title="ICH Detection",
    description=DICOM_DISCLAIMER,
)

if __name__ == "__main__":
    demo.launch()
