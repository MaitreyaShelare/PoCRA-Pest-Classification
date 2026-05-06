from PIL import Image
from pathlib import Path

root = Path("data/processed")

bad_files = []

for img_path in root.rglob("*"):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"BAD: {img_path}")
            print(e)
            bad_files.append(img_path)

print(f"\nTotal bad files: {len(bad_files)}")