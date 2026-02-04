import os
import shutil
from typing import Any, Dict, List, Tuple

import torch
from torchvision.io import decode_image, write_png
from tqdm import tqdm

from src.config.default import get_cfg
from src.utils import import_class_from_path


def get_file_id(file: str) -> str:
    return file.split(".")[0].split("_")[0]


def remove_and_create_folder(folder: str, delete_folder: bool) -> None:
    if delete_folder and os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def get_files(folder: str, extension: str) -> List[str]:
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(extension)]
    files.sort(key=get_file_id)
    return files


def get_wfdb(file_id: str) -> List[str]:
    file_id = file_id.split("-")[0]
    hea_file = f"{file_id}.hea"
    dat_file = f"{file_id}.dat"
    return [hea_file, dat_file]


def transform_tensors(scan: torch.Tensor, mask: torch.Tensor, transform: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    scan = scan.float() / 255.0
    mask = mask.float() / 255.0
    mask[0] = 1 - mask[1] - mask[2]  # Fill the red channel so that each pixel sums to 1
    if transform:
        scan, mask = transform(scan, mask)
    mask[0] = 0
    scan = (scan * 255.0).to(torch.uint8)
    mask = (mask * 255.0).to(torch.uint8)
    return scan, mask


def get_transform(config: Dict[Any, Any]) -> Any:
    if "TRANSFORM" not in config:
        raise ValueError("TRANSFORM key not found in config.")
    transform_config = config["TRANSFORM"]
    transform_class = import_class_from_path(transform_config["class_path"])
    return transform_class(**transform_config.get("KWARGS", {}))


def get_transform_folder(idx: int) -> str:
    if idx % 20 == 0:  # 1/20 to validation
        return "val"
    return "train"


if __name__ == "__main__":
    torch.manual_seed(42)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    DELETE_PREVIOUS_DST_FOLDERS = True
    if not DELETE_PREVIOUS_DST_FOLDERS:

        print(
            """
Set DELETE_PREVIOUS_DST_FOLDERS to True when generating the dataset to ensure that no entries from previous runs are
 left in the destination folders. Defaulting to False to prevent accidental deletion of data.
            """
        )

    # Change the following paths to match the location of the ECG dataset on
    # your local machine. The paths should be relative to the location of this
    # file or absolute.
    NUM_TRANSFORMS_PER_FILE = 1
    SRC_FOLDER = "/data/generated_images"
    DST_FOLDER = "/data/generated_images/transformed"
    APPLY_TRANSFORM = True
    IGNORE_TRANSFORM_FOLDERS = False

    TRANSFORM_FOLDERS = ["train", "val", "test"]
    NUM_CLASSES = 3

    config = get_cfg("../config/photo_transform.yml")
    transform = get_transform(config)

    if IGNORE_TRANSFORM_FOLDERS:
        remove_and_create_folder(DST_FOLDER, DELETE_PREVIOUS_DST_FOLDERS)
    else:
        for folder in TRANSFORM_FOLDERS:
            remove_and_create_folder(f"{DST_FOLDER}/{folder}", DELETE_PREVIOUS_DST_FOLDERS)

    masks_files = get_files(SRC_FOLDER, "_mask.png")

    for idx, mask_file in enumerate(tqdm(masks_files, desc="Generating transformed dataset")):
        file_id = get_file_id(mask_file)

        if IGNORE_TRANSFORM_FOLDERS:
            transform_folder = ""
        else:
            transform_folder = get_transform_folder(idx)

        for i in range(NUM_TRANSFORMS_PER_FILE):
            dst_file_id = f"{file_id}_T{i}"
            dst_mask_file = f"{dst_file_id}_mask.png"
            dst_scan_file = f"{dst_file_id}.png"

            mask: torch.Tensor = decode_image(f"{SRC_FOLDER}/{file_id}_mask.png", mode="RGB")
            scan: torch.Tensor = decode_image(f"{SRC_FOLDER}/{file_id}.png", mode="RGB")
            if APPLY_TRANSFORM:
                scan, mask = transform_tensors(scan, mask, transform)

            write_png(mask, f"{DST_FOLDER}/{transform_folder}/{dst_mask_file}", compression_level=0)
            write_png(scan, f"{DST_FOLDER}/{transform_folder}/{dst_scan_file}", compression_level=0)

            src_hea_file, src_dat_file = get_wfdb(file_id)
            # Copy the wfdb files to the destination folder
            shutil.copyfile(f"{SRC_FOLDER}/{src_hea_file}", f"{DST_FOLDER}/{transform_folder}/{dst_file_id}.hea")
            shutil.copyfile(f"{SRC_FOLDER}/{src_dat_file}", f"{DST_FOLDER}/{transform_folder}/{dst_file_id}.dat")

    print("Finished generating transformed ECG dataset.")
