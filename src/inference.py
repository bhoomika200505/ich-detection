import sys
import torch
from PIL import Image
import torchvision.transforms as T

from model import build_model
from dataset import LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def predict(image_path, weights_path="best_model.pt", threshold=0.5):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]

    result = {label: float(p) for label, p in zip(LABELS, probs)}
    flagged = [l for l, p in result.items() if p >= threshold]
    return result, flagged


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "subset_data/example.png"
    probs, flagged = predict(path)
    for label, p in probs.items():
        print(f"{label:20s} {p:.3f}")
    print("Flagged:", flagged or "none (no hemorrhage detected)")
