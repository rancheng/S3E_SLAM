import glob
import os
import shutil
from tqdm import tqdm

data_root = "/mnt/Data/Shared/data_generate/patch_image_show"

files = glob.glob(os.path.join(data_root, "*.png"))

for file in tqdm(files):
    fname = file.split("/")[-1]
    fcomponents = fname.split("_")
    fcomponents[0] = fcomponents[0].zfill(6)
    fname_updated = "_".join(fcomponents)
    updated_f = os.path.join(data_root, fname_updated)
    os.rename(file, updated_f)