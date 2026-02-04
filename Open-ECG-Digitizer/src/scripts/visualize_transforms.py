import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy import load

from src.transform.vision import ComposedTransform


class Visualizer:
    def __init__(self, n: int) -> None:
        """
        Initialize the visualizer.

        Args:
            n (int): Number of black-and-white and RGB images to generate.
        """
        self.n = n
        self.output_dir = "visualized_transforms"
        self.output_dir = self.create_output_dir(self.output_dir)

    def create_output_dir(self, dirname: str) -> str:
        """
        Create a directory beside the current script's file location.

        Args:
            dirname (str): Name of the directory to create.

        Returns:
            str: Absolute path to the created directory.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, dirname)

        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def _load_image_and_mask(self, image_path: str, mask_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load an image and its corresponding mask.

        Args:
            image_path (str): Path to the image file.
            mask_path (str): Path to the mask file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loaded image and mask tensors.
        """
        image = torch.tensor(plt.imread(image_path))
        mask = torch.tensor(load(mask_path))
        mask = nn.functional.one_hot(mask, num_classes=3).permute(2, 0, 1).float()
        return image, mask

    def _apply_transform(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the composed transformation to the image and mask.

        Args:
            image (torch.Tensor): The input image tensor.
            mask (torch.Tensor): The corresponding mask tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed image and mask tensors.
        """
        transform_config = [
            {"class_path": "src.transform.vision.GreyscaleTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomShiftTextTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomTextOverlayTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomFlipTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomBlurOrSharpnessTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomGammaTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomResizedCropTransform", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomRotation", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomJPEGCompression", "KWARGS": {}},
            {"class_path": "src.transform.vision.RandomGradientOverlay", "KWARGS": {}},
            {"class_path": "src.transform.vision.RefineMask", "KWARGS": {}},
        ]

        transform = ComposedTransform(transform_config)
        return transform(image, mask)  # type: ignore

    def _save_plot(self, image: torch.Tensor, mask: torch.Tensor, filename: str, greyscale: bool) -> None:
        """
        Save a plot of the image and mask.

        Args:
            image (torch.Tensor): The input image tensor.
            mask (torch.Tensor): The corresponding mask tensor.
            filename (str): Name of the file to save the plot.
            greyscale (bool): Whether the image is greyscale or RGB.
        """
        fig, ax = plt.subplots(2, 1, figsize=(20, 24))
        if greyscale:
            ax[0].imshow(image[0], cmap="gray", interpolation="none")
        else:
            ax[0].imshow(image.permute(1, 2, 0), interpolation="none")
        mask_plot = mask.clone()
        mask_plot[0] = 0.0
        ax[1].imshow(mask_plot.permute(1, 2, 0), interpolation="none")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def visualize(self, image_path: str, mask_path: str) -> None:
        """
        Generate and save visualizations for the given number of images.

        Args:
            image_path (str): Path to the image file.
            mask_path (str): Path to the mask file.
        """
        for i in range(self.n):
            # Load image and mask
            image, mask = self._load_image_and_mask(image_path, mask_path)

            # Generate and save original images
            greyscale = i % 2 == 0
            if greyscale:
                image = image.mean(2, keepdim=True).float().permute(2, 0, 1)
            else:
                image = image.permute(2, 0, 1).float()

            image_transformed, mask_transformed = self._apply_transform(image, mask)
            self._save_plot(image_transformed, mask_transformed, f"{i + 1}.png", greyscale)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and visualize image transformations.")
    parser.add_argument("-n", type=int, required=False, default=1, help="Number of images to generate.")
    args = parser.parse_args()

    visualizer = Visualizer(n=args.n)
    visualizer.visualize(
        image_path="test/test_data/data/ecg_scans/10_1.png",
        mask_path="test/test_data/data/ecg_masks/masks/10_1.npy",
    )
