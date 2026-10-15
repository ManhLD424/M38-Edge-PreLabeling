"""Dataset and weights downloader module."""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from concurrent.futures import ThreadPoolExecutor

BDD100K_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign"
]

MODELS = {
    "yolo26n.pt": "dronefreak/bdd100k-yolo26n",
    "yolo26s.pt": "dronefreak/bdd100k-yolo26s",
    "yolo26m.pt": "dronefreak/bdd100k-yolo26m",
}

def download_models(output_dir="models/baseline"):
    """Download pretrained YOLO26 baseline PyTorch checkpoints from Hugging Face."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    print("=== Downloading YOLO26 Checkpoints ===")
    
    downloaded_paths = {}
    for filename, repo_id in MODELS.items():
        dest = out_path / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[{filename}] already exists ({dest.stat().st_size / (1024*1024):.2f} MB), skipping.")
            downloaded_paths[filename] = dest
            continue
        print(f"Downloading {filename} from {repo_id}...")
        downloaded = hf_hub_download(repo_id=repo_id, filename="best.pt")
        shutil.copy(downloaded, dest)
        print(f"-> Saved {filename} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        downloaded_paths[filename] = dest
        
    return downloaded_paths

def _download_file_pair(img_rel, lbl_rel, img_dir, lbl_dir):
    try:
        img_hf = hf_hub_download(repo_id="dronefreak/BDD100K", repo_type="dataset", filename=img_rel)
        img_name = Path(img_rel).name
        shutil.copy(img_hf, img_dir / img_name)

        lbl_hf = hf_hub_download(repo_id="dronefreak/BDD100K", repo_type="dataset", filename=lbl_rel)
        lbl_name = Path(lbl_rel).name
        shutil.copy(lbl_hf, lbl_dir / lbl_name)
        return True
    except Exception as e:
        print(f"Error downloading {img_rel}: {e}")
        return False

def download_val_dataset(output_dir="data/bdd100k_val", num_samples=200):
    """Download curated validation subset from Hugging Face BDD100K dataset."""
    data_path = Path(output_dir).resolve()
    img_dir = data_path / "images" / "val"
    lbl_dir = data_path / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading BDD100K Validation Sample ({num_samples} images & labels) ===")
    api = HfApi()
    files = api.list_repo_files(repo_id="dronefreak/BDD100K", repo_type="dataset")

    img_files = sorted([f for f in files if f.startswith("data/images/test/shard_000/") and f.endswith(".jpg")])
    lbl_files_set = set(files)

    pairs_to_download = []
    for img_f in img_files:
        lbl_f = img_f.replace("data/images/", "data/labels/").replace(".jpg", ".txt")
        if lbl_f in lbl_files_set:
            pairs_to_download.append((img_f, lbl_f))
        if len(pairs_to_download) >= num_samples:
            break

    print(f"Found {len(pairs_to_download)} paired samples. Starting parallel download...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda p: _download_file_pair(p[0], p[1], img_dir, lbl_dir), pairs_to_download))

    success_count = sum(1 for r in results if r)
    print(f"Successfully downloaded {success_count}/{len(pairs_to_download)} image-label pairs into {data_path}")

    write_dataset_yaml(data_path)
    return data_path

def write_dataset_yaml(data_dir):
    """Generate YOLO-compatible dataset configuration YAML."""
    data_path = Path(data_dir).resolve()
    yaml_path = data_path / "bdd100k_val.yaml"
    names_block = "\n".join([f"  {idx}: {cls}" for idx, cls in enumerate(BDD100K_CLASSES)])
    yaml_content = f"""# BDD100K Validation Subset for YOLO Evaluation
path: {data_path.as_posix()}
train: images/val
val: images/val
test: images/val

nc: {len(BDD100K_CLASSES)}
names:
{names_block}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Wrote dataset config to: {yaml_path}")
    return yaml_path
