import os

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image

from src.model.cropper import Cropper
from src.model.perspective_detector import PerspectiveDetector


def show_image(image: torch.Tensor, save_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.imshow(image.permute(1, 2, 0), interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, image.shape[2])
    ax.set_ylim(image.shape[1], 0)
    plt.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.show()


def main() -> None:
    perspective_detector = PerspectiveDetector(num_thetas=150)
    cropper = Cropper()

    image_paths = [
        "/data/validation_images/IMG20241203092505.jpg",
        "/data/validation_images/IMG20241226080324.jpg",
        "/data/validation_images/IMG20250103161210.jpg",
    ]

    os.makedirs("src/report/figures/perspective", exist_ok=True)

    for i, image_path in enumerate(image_paths, start=1):

        image = read_image(image_path).float().div(255)
        params = perspective_detector(image.cuda())
        src_points = cropper((image.mean(0) < image.mean() / 2).float(), params)
        resampled = cropper.apply_perspective(image, src_points, 0)

        show_image(image, f"src/report/figures/perspective/{i}_raw.png")
        show_image(resampled.cpu(), f"src/report/figures/perspective/{i}_processed.png")


if __name__ == "__main__":
    main()
