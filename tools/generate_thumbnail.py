"""Use an actual Studio screenshot as the project thumbnail, without invented metrics."""
import argparse
import shutil
from pathlib import Path
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--screenshot',required=True,type=Path)
args=parser.parse_args()
with Image.open(args.screenshot) as image:
    image.verify()
target=Path(__file__).resolve().parents[1]/'docs/figures/thumbnail.png'
target.parent.mkdir(exist_ok=True)
shutil.copy2(args.screenshot,target)
print(f'Copied actual UI screenshot: {target}')
