import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_directories(highlight_folder: str, save_folder: str) -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(highlight_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)


def load_image_as_array(path: str) -> np.ndarray[Any, Any]:
    """Load image and return as NumPy array."""
    return np.array(Image.open(path), dtype=np.int64)


def save_image_from_array(array: np.ndarray[Any, Any], path: str) -> None:
    """Save NumPy array as image."""
    Image.fromarray(array.astype(np.uint8)).save(path)


def compute_redaction_mask(
    image_array: np.ndarray[Any, Any], redaction_color: np.ndarray[Any, Any], threshold: float
) -> np.ndarray[Any, Any]:
    """
    Compute a boolean mask for pixels that match the redaction color within a norm threshold.
    """
    diff = image_array - redaction_color.reshape(1, 1, 3)
    norm = np.linalg.norm(diff, axis=-1)
    return norm < threshold  # type: ignore


def visualize_histogram(norm_values: np.ndarray[Any, Any]) -> None:
    """Optional: display a histogram of pixel value distances."""
    plt.figure()
    plt.hist(norm_values.flatten(), bins=256, color="green", alpha=0.7)
    plt.title("Histogram of Redacted Image Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("redacted_image_histogram.png")


def process_images(
    original_folder: str,
    redacted_folder: str,
    highlight_folder: str,
    save_folder: str,
    redaction_color: np.ndarray[Any, Any],
    redaction_threshold: float,
    output_color: np.ndarray[Any, Any],
    debug: bool = False,
) -> None:
    """Main processing loop for redacting images."""
    ensure_directories(highlight_folder, save_folder)
    image_names = tqdm(os.listdir(redacted_folder), desc="Redacting images", unit="image")

    for image_name in image_names:
        original_path = os.path.join(original_folder, image_name)
        redacted_path = os.path.join(redacted_folder, image_name)

        original_img = load_image_as_array(original_path)
        redacted_img = load_image_as_array(redacted_path)

        redaction_mask = compute_redaction_mask(redacted_img, redaction_color, redaction_threshold)

        if debug:
            norm_values = np.linalg.norm(redacted_img - redaction_color.reshape(1, 1, 3), axis=-1)
            visualize_histogram(norm_values)

        # Save visualization version (green redaction)
        highlighted_img = original_img.copy()
        highlighted_img[redaction_mask] = redaction_color
        save_image_from_array(highlighted_img, os.path.join(highlight_folder, image_name))

        # Save final dataset version (black redaction)
        redacted_output = original_img.copy()
        redacted_output[redaction_mask] = output_color
        save_image_from_array(redacted_output, os.path.join(save_folder, image_name))


if __name__ == "__main__":
    # Configuration
    ORIGINAL_IMAGE_FOLDER = "/home/datasets/ecg-digitization/digital_redaction/elias_phone_all"
    REDACTED_FOLDER = "/home/datasets/ecg-digitization/digital_redaction/elias_all_redacted_final"
    REDACTION_HIGHLIGHTED_SAVE_FOLDER = "/home/datasets/ecg-digitization/redacted/elias_highlighted_digital_redactions"
    REDACTION_SAVE_FOLDER = "/home/datasets/ecg-digitization/redacted/elias_digital_redactions"

    REDACTION_COLOR = np.array([0, 255, 0])
    REDACTION_NORM_THRESHOLD = 50.0
    OUTPUT_REDACTION_COLOR = np.array([0, 0, 0])
    DEBUG = True

    process_images(
        original_folder=ORIGINAL_IMAGE_FOLDER,
        redacted_folder=REDACTED_FOLDER,
        highlight_folder=REDACTION_HIGHLIGHTED_SAVE_FOLDER,
        save_folder=REDACTION_SAVE_FOLDER,
        redaction_color=REDACTION_COLOR,
        redaction_threshold=REDACTION_NORM_THRESHOLD,
        output_color=OUTPUT_REDACTION_COLOR,
        debug=DEBUG,
    )
