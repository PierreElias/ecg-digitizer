import math
import os
import random
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T2
from torch import nn
from torchvision.io import read_image
from torchvision.transforms.functional import perspective

from src.utils import import_class_from_path


class RandomShiftTextTransform(nn.Module):
    """Randomly overlays a shifted version of the text in the image."""

    def __init__(
        self,
        roll_x_range: Tuple[int, int] = (-1000, 1000),
        roll_y_range: Tuple[int, int] = (-1000, 1000),
        opacity_range: Tuple[float, float] = (0.4, 0.8),
    ):
        super().__init__()
        self.roll_x_range = roll_x_range
        self.roll_y_range = roll_y_range
        self.opacity_range = opacity_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        roll_x = random.randint(*self.roll_x_range)
        roll_y = random.randint(*self.roll_y_range)
        opacity = random.uniform(*self.opacity_range)

        text_mask = (mask[1:2] > 0).to(img.dtype)

        text_region = img * text_mask

        rolled_text_mask = torch.roll(text_mask, shifts=(roll_x, roll_y), dims=(-2, -1))
        rolled_text_region = torch.roll(text_region, shifts=(roll_x, roll_y), dims=(-2, -1))

        blended_img = torch.where(rolled_text_mask.bool(), (1 - opacity) * img + opacity * rolled_text_region, img)

        background_mask = (mask[2:] == 0).bool()
        mask[1] = torch.where(
            background_mask & (rolled_text_mask[0] > 0), torch.max(opacity * torch.ones_like(mask[1]), mask[1]), mask[1]
        )[0]
        mask[0] = torch.where(
            background_mask & (rolled_text_mask[0] > 0),
            torch.min((1 - opacity) * torch.ones_like(mask[0]), mask[0]),
            mask[0],
        )[0]

        return blended_img, mask


class RandomTextOverlayTransform(nn.Module):
    """Reads a set of text images and overlays them on the input image."""

    def __init__(self, text_path: Union[str | None] = None, opacity_range: Tuple[float, float] = (0.1, 0.8)):
        super().__init__()
        self.text_path = text_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay_images")
        self.texts = [
            torch.tensor(plt.imread(os.path.join(self.text_path, f))).float().permute(2, 0, 1)[:3]
            for f in os.listdir(self.text_path)
        ]
        self.opacity_range = opacity_range
        self.rotation = T.RandomRotation(degrees=180, fill=(1,), interpolation=T.InterpolationMode.BILINEAR)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text = None
        while text is None or text.shape[-1] < img.shape[-1] or text.shape[-2] < img.shape[-2]:
            text = self.texts[random.randint(0, len(self.texts) - 1)]
        text = self.rotation(text)

        if img.shape[0] == 1:
            text = text.mean(0, keepdim=True)

        c1 = random.randint(0, text.shape[-2] - img.shape[-2])
        c2 = random.randint(0, text.shape[-1] - img.shape[-1])
        text_crop = text[:, c1 : c1 + img.shape[-2], c2 : c2 + img.shape[-1]]

        binary_text_mask = (text_crop < 0.8).sum(0) > 0
        opacity = random.uniform(*self.opacity_range)

        img = torch.where(binary_text_mask, opacity * (text_crop) + (1 - opacity) * img, img)

        mask_channel_1_condition = binary_text_mask & (mask[2:] == 0).all(0)
        mask[1] = torch.where(mask_channel_1_condition, torch.max(opacity * torch.ones_like(mask[1]), mask[1]), mask[1])
        mask[0] = torch.where(
            mask_channel_1_condition, torch.min((1 - opacity) * torch.ones_like(mask[1]), mask[0]), mask[0]
        )

        return img, mask.float()


class RandomPerspectiveWithImageTransform(nn.Module):
    def __init__(self, image_path: str, distortion_scale: float = 0.15) -> None:
        super().__init__()
        self.image_path = image_path
        self.distortion_scale = distortion_scale
        self.image_paths = os.listdir(image_path)
        assert (
            len(self.image_paths) > 0
        ), f"No images found in {image_path}. Please check the path in transform config file."

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        index: int = int(torch.randint(0, len(self.image_paths), (1,)).item())
        random_image_path = self.image_paths[index]
        random_image = read_image(os.path.join(self.image_path, random_image_path)).float().div(255).clamp(0, 1)

        startpoints = [[0, 0], [img.shape[2], 0], [img.shape[2], img.shape[1]], [0, img.shape[1]]]
        offset_range = 0.5 * self.distortion_scale * img.shape[2]
        endpoints = torch.tensor(startpoints) + torch.rand((4, 2)).mul(2 * offset_range) - offset_range / 2
        midpoint = torch.tensor([img.shape[2] / 2, img.shape[1] / 2])
        endpoints = endpoints * 0.95 + midpoint[None, :] * 0.05

        fill_value = torch.rand((1,)).item()

        img = perspective(img.unsqueeze(0), startpoints, endpoints, fill=fill_value).squeeze(0)
        mask = perspective(mask.unsqueeze(0), startpoints, endpoints, fill=0.0).squeeze(0)

        if random_image.shape[1] <= img.shape[1] or random_image.shape[2] <= img.shape[2]:
            random_image = F.resize(random_image, (img.shape[1] + 5, img.shape[2] + 5))

        c11: int = int(torch.randint(0, random_image.shape[1] - img.shape[1], (1,)).item())
        c12: int = c11 + img.shape[1]
        c21: int = int(torch.randint(0, random_image.shape[2] - img.shape[2], (1,)).item())
        c22: int = c21 + img.shape[2]
        random_image = random_image[:, c11:c12, c21:c22]

        img[img == fill_value] = random_image[img == fill_value]
        mask_is_transformed = (mask == 0.0).sum(0) == 3
        mask[0][mask_is_transformed] = 0.5
        mask[1][mask_is_transformed] = 0.5
        mask[2][mask_is_transformed] = 0.0

        return img, mask


class RandomFlipTransform(nn.Module):
    """Randomly flips the image and mask."""

    def __init__(self, p_ud: float = 0.5, p_lr: float = 0.5, p_tr: float = 0.5):
        super().__init__()
        self.p_ud = p_ud
        self.p_lr = p_lr
        self.p_tr = p_tr

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p_ud:
            img, mask = F.vflip(img), F.vflip(mask)
        if random.random() < self.p_lr:
            img, mask = F.hflip(img), F.hflip(mask)
        if random.random() < self.p_tr:
            img, mask = img.transpose(-1, -2), mask.transpose(-1, -2)
        return img, mask


class GaussianBlurTransform(nn.Module):
    """Applies Gaussian blur to the image."""

    def __init__(self, kernel_size: int = 7, sigma: Tuple[float, float] = (0.0, 10.0), p: float = 0.1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma
        self.p = p

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask
        sigma = random.uniform(*self.sigma_range)
        img = F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigma)
        return img, mask


class RandomSharpnessTransform(nn.Module):
    """Adjusts the sharpness of the image."""

    def __init__(self, sharpness_range: Tuple[float, float] = (1.0, 3.0)):
        super().__init__()
        self.sharpness_range = sharpness_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sharpness = random.uniform(*self.sharpness_range)
        img = F.adjust_sharpness(img, sharpness_factor=sharpness)
        return img, mask


class RandomBlurOrSharpnessTransform(nn.Module):
    """Randomly applies either Gaussian blur or sharpness adjustment."""

    def __init__(
        self,
        p: float = 0.5,
        sharpness_range: Tuple[float, float] = (1.0, 2.0),
        kernel_size: int = 7,
        sigma: Tuple[float, float] = (0.0, 1.5),
    ):
        super().__init__()
        self.p = p
        self.blur = GaussianBlurTransform(kernel_size=kernel_size, sigma=sigma)
        self.sharpness = RandomSharpnessTransform(sharpness_range=sharpness_range)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            img, mask = self.blur(img, mask)
        else:
            img, mask = self.sharpness(img, mask)
        return img, mask


class RandomGammaTransform(nn.Module):
    """Applies gamma correction to the image."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = random.uniform(*self.gamma_range)
        img = F.adjust_gamma(img, gamma=gamma)
        return img, mask


class RandomResizedCropTransform(nn.Module):
    """Crops and resamples to a random size within specified height and width ranges."""

    def __init__(
        self,
        h_range: Tuple[int, int] = (1360, 2550),
        w_range: Tuple[int, int] = (1760, 3300),
        scale: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.h_range = h_range
        self.w_range = w_range
        self.scale = scale

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_height = random.randint(*self.h_range)
        target_width = random.randint(*self.w_range)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=self.scale, ratio=(target_width / target_height, target_width / target_height)
        )
        img = F.resized_crop(img, i, j, h, w, (target_height, target_width))
        mask = F.resized_crop(mask, i, j, h, w, (target_height, target_width))
        return img, mask


class RandomRotation(nn.Module):
    """Randomly rotates both image and mask."""

    def __init__(self, degrees: Union[float, Tuple[float, float]] = 10):
        super().__init__()
        self.degrees = degrees

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = (
            random.uniform(-self.degrees, self.degrees)
            if isinstance(self.degrees, (int, float))
            else random.uniform(*self.degrees)
        )
        img = F.rotate(img, angle, fill=1, interpolation=T.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, fill=[1, 0, 0], interpolation=T.InterpolationMode.BILINEAR)
        return img, mask


class RandomJPEGCompression(nn.Module):
    """Simulates JPEG compression artifacts."""

    def __init__(self, quality: Union[int, Tuple[int, int]] = (5, 100)):
        super().__init__()
        self.jpeg_transform = T2.JPEG(quality=quality)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = (img * 255).clamp(0, 255).byte()
        img = self.jpeg_transform(img)
        img = img.float() / 255
        return img, mask


class RandomCrop(nn.Module):
    """Randomly crops the image and mask to a specified size."""

    def __init__(self, size: Tuple[int, int] = (1024, 1024)):
        super().__init__()
        self.size = size

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop the image and mask to the specified size."""
        i, j, h, w = self.get_params(img)
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return img, mask

    def get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get parameters for a random crop."""
        i = random.randint(0, img.shape[1] - self.size[0])
        j = random.randint(0, img.shape[2] - self.size[1])
        h, w = self.size
        return i, j, h, w


class RandomSaturationContrast(nn.Module):
    def __init__(self, sat_factor: float = 2.5, contrast_factor: float = 2.5):
        super().__init__()
        self.sat_factor = sat_factor
        self.contrast_factor = contrast_factor

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sat factor is a random number between 1 and self.sat_factor
        sat_factor = random.uniform(1.0, self.sat_factor)
        contrast_factor = random.uniform(1.0, self.contrast_factor)

        # Compute grayscale image
        gray = img.mean(dim=-3, keepdims=True)  # type: ignore

        # Adjust saturation
        img = gray + (img - gray) * sat_factor
        img = torch.clamp(img, 0.0, 1.0)

        # Adjust contrast
        img = (img - 0.5) * contrast_factor + 0.5
        img = torch.clamp(img, 0.0, 1.0)

        return img, mask


class RandomLine(nn.Module):
    """Randomly draws a solid color line over the image tensor."""

    def __init__(self, p: float = 0.5, max_line_width: int = 50):
        super().__init__()
        self.p = p
        self.max_line_width = max_line_width

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask

        C, H, W = img.shape
        device = img.device

        # Random angle and center point
        theta = random.uniform(0, 2 * math.pi)
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)

        # Line extent
        line_len = int(math.hypot(H, W))
        dx = int(math.cos(theta) * line_len)
        dy = int(math.sin(theta) * line_len)

        x0 = cx - dx // 2
        y0 = cy - dy // 2
        x1 = cx + dx // 2
        y1 = cy + dy // 2

        # Generate coordinate grid
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

        # Vector math
        line_vec = torch.tensor([x1 - x0, y1 - y0], device=device, dtype=torch.float32)
        line_len = torch.norm(line_vec)
        if line_len == 0:
            return img, mask  # Degenerate line

        px = xx - x0
        py = yy - y0
        cross = line_vec[1] * px - line_vec[0] * py
        dist = torch.abs(cross) / line_len

        # Only within line segment
        dot = (px * line_vec[0] + py * line_vec[1]) / (line_len**2)
        within_segment = (dot >= 0) & (dot <= 1)

        # Line width mask
        line_width = random.randint(1, self.max_line_width)
        mask_line = (dist <= (line_width / 2)) & within_segment  # (H, W)

        # Assign random color directly to affected pixels
        color = torch.rand(C, device=device).unsqueeze(-1) * 0.2
        mask_fill = torch.tensor([0.5, 0.5, 0]).unsqueeze(-1).to(device)
        img[:, mask_line] = color
        mask[:, mask_line] = mask_fill

        return img, mask


class RandomQRCode(nn.Module):
    """Randomly overlays a (fake) QR code on the image."""

    def __init__(self, p: float = 0.05, qr_size: int = 20):
        super().__init__()
        self.p = p
        self.qr_size = qr_size
        self.max_qr_size = qr_size * 16

    def _generate_qr_code(self) -> torch.Tensor:
        qr = torch.zeros((self.qr_size, self.qr_size), dtype=torch.float32)
        qr_vals = torch.rand(self.qr_size, self.qr_size) > 0.5
        qr[qr_vals] = 1.0
        return qr

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask
        qr = self._generate_qr_code()
        qr_size = random.randint(self.qr_size, self.max_qr_size)
        qr = torch.nn.functional.interpolate(qr.unsqueeze(0).unsqueeze(0), (qr_size, qr_size), mode="nearest").squeeze(
            0
        )
        h, w = img.shape[1], img.shape[2]
        c1 = random.randint(0, h - qr_size)
        c2 = random.randint(0, w - qr_size)
        img[:, c1 : c1 + qr_size, c2 : c2 + qr_size] = qr.unsqueeze(0)
        mask[0, c1 : c1 + qr_size, c2 : c2 + qr_size] = 0.0
        mask[1, c1 : c1 + qr_size, c2 : c2 + qr_size] = 1 - qr
        mask[2, c1 : c1 + qr_size, c2 : c2 + qr_size] = 0.0
        return img, mask


class RandomBorder(nn.Module):
    """Randomly adds a border to the image."""

    def __init__(self, p: float = 0.1, max_border_size: int = 90):
        super().__init__()
        self.p = p
        self.max_border_size = max_border_size

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask
        border_size: int = int(torch.randint(1, self.max_border_size + 1, (1,)).item())
        border_color = torch.rand(3).to(img.device).unsqueeze(1).unsqueeze(2)
        mask_fill = torch.tensor([0.0, 1.0, 0]).to(img.device).unsqueeze(1).unsqueeze(2)

        img[:, :border_size, :] = border_color
        img[:, -border_size:, :] = border_color
        img[:, :, :border_size] = border_color
        img[:, :, -border_size:] = border_color

        mask[:, :border_size, :] = mask_fill
        mask[:, -border_size:, :] = mask_fill
        mask[:, :, :border_size] = mask_fill
        mask[:, :, -border_size:] = mask_fill

        return img, mask


class RandomFourierDropout(nn.Module):
    def __init__(self, p: float = 0.1, max_dropout_rate: float = 0.5):
        super().__init__()
        self.p = p
        self.max_dropout_rate = max_dropout_rate

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) > self.p:
            return img, mask

        dropout_rate = torch.rand(1) * self.max_dropout_rate

        img_fft = torch.fft.fft2(img)
        dropout_mask = torch.rand_like(img_fft.real) > dropout_rate
        dropout_mask[:50, :50] = 1
        dropout_mask[-50:, :50] = 1

        transformed_img = torch.fft.ifft2(img_fft * dropout_mask).abs()
        transformed_img = (transformed_img - transformed_img.min()) / (transformed_img.max() - transformed_img.min())

        return transformed_img.clamp(0, 1), mask


class RandomGradientOverlay(nn.Module):
    """Applies a gradient overlay to simulate uneven lighting."""

    def __init__(self, p: float = 0.1, opacity_range: Tuple[float, float] = (-0.3, 0.3)):
        super().__init__()
        self.p = p
        self.opacity_range = opacity_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = img.shape[1], img.shape[2]
        if random.random() < self.p:
            gradient = torch.linspace(0, 1, w).unsqueeze(0)
            gradient = gradient.repeat(h, 1).unsqueeze(0)  # (1, H, W)
        else:
            gradient = torch.linspace(0, 1, h).unsqueeze(1)
            gradient = gradient.repeat(1, w).unsqueeze(0)  # (1, H, W)

        opacity = random.uniform(*self.opacity_range)

        img = img + opacity * gradient.to(img.device)
        img = img.clamp(0, 1)
        return img, mask


class RefineMask(nn.Module):
    """Ensures that the mask is valid after transformations."""

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask[0] /= mask[0].max().clamp(min=1e-6)
        mask[2] /= mask[2].max().clamp(min=1e-6)
        signal_mask = mask[2] > 0
        mask[0] = torch.where(signal_mask, torch.zeros_like(mask[0]), mask[0])
        mask[1] = torch.where(signal_mask, torch.zeros_like(mask[1]), mask[1])
        return img, mask


class ComposedTransform(nn.Module):
    """Applies transformations in sequence."""

    def __init__(self, transform_config: List[Dict[Any, Any]]):
        """
        Initializes a composed transform based on the given configuration.

        Args:
            transforms (dict): Configuration containing class path and transforms.
        """
        super(ComposedTransform, self).__init__()
        transforms = []
        for transform_def in transform_config:
            transform_class = import_class_from_path(transform_def["class_path"])
            transform = transform_class(**transform_def.get("KWARGS", {}))
            transforms.append(transform)
        self.transforms = transforms

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            img, mask = transform.forward(img, mask)
        return img, mask


class GreyscaleTransform(nn.Module):
    """Converts the image to greyscale."""

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img, mask


class RandomJPEGCompressionTransform(nn.Module):
    """Applies random JPEG compression to the image."""

    def __init__(self, quality: Union[int, Tuple[int, int]] = (2, 98), p: float = 0.5):
        super().__init__()
        self.jpeg_transform = T2.JPEG(quality=quality)
        self.p = p

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask
        img = (img * 255).clamp(0, 255).byte()
        img = self.jpeg_transform(img)
        img = img.float() / 255
        return img, mask


class RandomZoomTransform(nn.Module):

    def __init__(self, scale_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.2):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.p:
            return img, mask
        orig_height, orig_width = img.shape[1], img.shape[2]
        scale = random.uniform(*self.scale_range)

        # If scale == 1.0, return original
        if scale == 1.0:
            return img, mask

        # Resize (zoom in)
        new_height = int(orig_height * scale)
        new_width = int(orig_width * scale)
        img = F.resize(img, [new_height, new_width])
        mask = F.resize(mask, [new_height, new_width])

        # Center crop to original size
        top = (new_height - orig_height) // 2
        left = (new_width - orig_width) // 2
        img = F.crop(img, top, left, orig_height, orig_width)
        mask = F.crop(mask, top, left, orig_height, orig_width)

        return img, mask
