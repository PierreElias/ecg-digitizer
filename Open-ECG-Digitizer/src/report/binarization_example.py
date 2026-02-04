import matplotlib.pyplot as plt
import mpl_toolkits  # type: ignore
import numpy as np
import torch
from skimage.feature import peak_local_max
from torch.nn.functional import max_pool2d
from torchvision.io import read_image

from src.model.perspective_detector import PerspectiveDetector

LARGEFONT = 20
CMAP = "grey"
SZ = 200
YSTART = 1300
XSTART = 2100

plt.rcParams["font.family"] = "serif"


def main() -> None:
    # Load process image
    image_path = "/data/validation_images/IMG20241226080324.jpg"
    image = read_image(image_path)

    pd = PerspectiveDetector(num_thetas=400)
    binary_image = ~pd.binarize(image.cuda()).cpu()
    thetas = torch.linspace(-np.pi / 4, 3 * np.pi / 4, 400)
    hough_domain, _ = pd.hough_transform(~binary_image, thetas)

    hough_domain = max_pool2d(hough_domain[None, None], kernel_size=(5, 1), stride=(5, 1), padding=(2, 0)).squeeze()

    num_rhos = hough_domain.shape[0]
    hough_domain = hough_domain[num_rhos // 4 : 3 * num_rhos // 4]
    variances = pd.calculate_line_variances(hough_domain.float())

    peaks = peak_local_max(variances.numpy(), min_distance=10, num_peaks=2)  # type: ignore

    # Plot the image, binary image, hough domain and angle-angle domain
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))
    axs[0, 0].imshow(image.permute(1, 2, 0))
    axs[0, 0].set_title("Original Image", fontsize=LARGEFONT, pad=10)

    axs[0, 1].imshow(binary_image, cmap=CMAP, origin="upper")

    part_of_binary_image = binary_image[YSTART : YSTART + SZ, XSTART : XSTART + SZ]
    binary_image_copy = binary_image.clone()
    # reverse along axis 0
    binary_image_copy[YSTART : YSTART + SZ, XSTART : XSTART + SZ] = part_of_binary_image.flip(0)
    # Create an inset axes
    ax_inset = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(
        axs[0, 1], zoom=6, loc="upper right", borderpad=0.5
    )
    ax_inset.imshow(binary_image_copy, cmap=CMAP, interpolation="nearest", origin="upper")
    ax_inset.set_xlim(XSTART, XSTART + SZ)
    ax_inset.set_ylim(YSTART, YSTART + SZ)
    # set border color of ax_inset to red
    ax_inset.spines["bottom"].set_color("C3")
    ax_inset.spines["top"].set_color("C3")
    ax_inset.spines["right"].set_color("C3")
    ax_inset.spines["left"].set_color("C3")
    mpl_toolkits.axes_grid1.mark_inset(axs[0, 1], ax_inset, loc1=4, loc2=2, fc="none", ec="C3")
    axs[0, 1].set_title("Binary Image", fontsize=LARGEFONT, pad=10)

    # fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    axs[1, 0].imshow(-hough_domain, cmap=CMAP, origin="upper", aspect="auto")
    axs[1, 0].set_ylabel(r"$\rho$", fontsize=LARGEFONT)
    axs[1, 0].set_xlabel(r"$\theta$", fontsize=LARGEFONT)

    # plot the line with x=peaks[0, 1] and x=peaks[1, 1], y being the top and bottom values
    axs[1, 0].plot([peaks[0, 0], peaks[0, 1]], [0, hough_domain.shape[0]], "C0", linestyle=":")
    axs[1, 0].plot([peaks[1, 0], peaks[1, 1]], [0, hough_domain.shape[0]], "C0", linestyle=":")
    axs[1, 0].set_xlim(0, hough_domain.shape[1] - 1)
    axs[1, 0].set_ylim(hough_domain.shape[0], 0)
    axs[1, 0].set_title("Angle-Radius Domain", fontsize=LARGEFONT, pad=10)

    axs[1, 1].imshow(-variances, cmap=CMAP, origin="upper", aspect="auto")
    axs[1, 1].set_ylabel(r"$\theta_{top}$", fontsize=LARGEFONT)
    axs[1, 1].set_xlabel(r"$\theta_{bot}$", fontsize=LARGEFONT)

    axs[1, 1].scatter(peaks[:, 1], peaks[:, 0], c="C0", s=500, marker="*", alpha=0.7)
    axs[1, 1].set_title("Angle-Angle Domain", fontsize=LARGEFONT, pad=10)

    for ax in [ax_inset, *axs.flatten()]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("src/report/figures/perspective/perspective_domains.pgf", bbox_inches="tight", dpi=200)

    # Change relative paths to match Overleaf
    with open("src/report/figures/perspective/perspective_domains.pgf", "r") as f:
        content = f.read()
    updated_content = content.replace("perspective_domains-img", "figures/perspective/perspective_domains-img")
    with open("src/report/figures/perspective/perspective_domains.pgf", "w") as f:
        f.write(updated_content)

    plt.show()


if __name__ == "__main__":
    main()
