import os
from typing import Any, List, Tuple, Union

import torch
from torchvision.io import decode_image


class ECGScanDataset(torch.utils.data.Dataset[Any]):
    def __init__(self, data_path: str, transform: Any = None, load_n: Union[None, int] = None) -> None:
        self.data_path = data_path
        self.transform = transform

        # Prepend current software directory if path is not absolute.
        if not os.path.isabs(self.data_path):
            self.data_path = os.path.join(os.path.dirname(__file__), self.data_path)

        self.ecg_scan_files, self.ecg_mask_files = self._find_ecg_files()
        if load_n:
            self.ecg_scan_files = self.ecg_scan_files[:load_n]
            self.ecg_mask_files = self.ecg_mask_files[:load_n]

    def _find_ecg_files(self) -> Tuple[List[str], List[str]]:
        if not os.path.exists(self.data_path):
            raise ValueError(f"Path {self.data_path} does not exist.")

        def is_scan_file(file: str) -> bool:
            return file.endswith(".png") and not file.endswith("_mask.png")

        ecg_scan_filenames = list(map(lambda x: x.split(".")[0], filter(is_scan_file, os.listdir(self.data_path))))

        ecg_scan_files = list(map(lambda x: os.path.join(self.data_path, f"{x}.png"), ecg_scan_filenames))
        ecg_mask_files = list(map(lambda x: os.path.join(self.data_path, f"{x}_mask.png"), ecg_scan_filenames))
        return ecg_scan_files, ecg_mask_files

    def _load_scan(self, file: str) -> torch.Tensor:
        scan: torch.Tensor = decode_image(os.path.join(self.data_path, file), mode="RGB").float() / 255.0
        return scan

    def _load_mask(self, file: str) -> torch.Tensor:
        mask: torch.Tensor = decode_image(os.path.join(self.data_path, file), mode="RGB").float() / 255.0
        return mask

    def __len__(self) -> int:
        return len(self.ecg_scan_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scan = self._load_scan(self.ecg_scan_files[idx])
        mask = self._load_mask(self.ecg_mask_files[idx])

        if self.transform:
            scan, mask = self.transform(scan, mask)

        return scan, mask
