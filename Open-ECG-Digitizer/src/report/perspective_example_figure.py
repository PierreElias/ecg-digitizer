import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

CANVAS_SIZE = (1700, 2200)
SRC_POINT_H = (-100, 4_500)
SRC_POINT_V = (5_000, 300)
NUM_LINES = 6

MARGIN = 300
H_PTS = np.linspace(MARGIN, CANVAS_SIZE[1] - MARGIN, NUM_LINES)
V_PTS = np.linspace(MARGIN, CANVAS_SIZE[0] - MARGIN, NUM_LINES)

COLOR_V = "C0"
COLOR_H = "C3"
LINEWIDTH = 1.5
GREYCOLOR = "#000000"
BIGSIZE = 26
NORMALSIZE = 20

plt.rcParams["font.family"] = "serif"


def get_intercept_slope(p1, p2):  # type: ignore
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return intercept, slope


def get_lines(H_PTS=H_PTS, V_PTS=V_PTS):  # type: ignore
    lines = []
    for h in H_PTS:
        intercept, slope = get_intercept_slope(SRC_POINT_H, (h, 0))  # type: ignore
        lines.append((intercept, slope))
    for v in V_PTS:
        intercept, slope = get_intercept_slope(SRC_POINT_V, (0, v))  # type: ignore
        lines.append((intercept, slope))
    return lines


def get_rho_theta(lines):  # type: ignore
    rho_theta = []
    for intercept, slope in lines:
        theta = np.arctan(-1 / slope)
        rho = np.abs(intercept / np.sqrt(1 + slope**2))
        rho_theta.append((rho, theta))
    return rho_theta


def main() -> None:
    lines = get_lines()  # type: ignore
    rho_theta = get_rho_theta(lines)  # type: ignore
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    axs[0].set_xlim(0, CANVAS_SIZE[1])
    axs[0].set_ylim(0, CANVAS_SIZE[0])
    for i, (intercept, slope) in enumerate(lines):
        if i == NUM_LINES - 2:
            # find the line that goes through the origin and is perpendicular to the last line
            slope_perp = -1 / slope
            intercept_perp = 0
            rho, theta = rho_theta[i]
            x_max = rho * np.cos(theta)
            x = np.linspace(0, x_max, 5)
            y = slope_perp * x + intercept_perp
            axs[0].plot(x, y, color=GREYCOLOR, linestyle="--", linewidth=LINEWIDTH)

            # Use Arc to draw the angle between the x axis and the dotted line
            arc = Arc((0, 0), 800, 800, theta1=0, theta2=np.degrees(theta), color=GREYCOLOR, linewidth=LINEWIDTH)
            axs[0].add_patch(arc)
            axs[0].text(420, 50, r"$\theta_i$", fontsize=BIGSIZE, color=GREYCOLOR)
            axs[0].text(650, 320, r"$\rho_i$", fontsize=BIGSIZE, color=GREYCOLOR)
        x = np.linspace(-3e3, 3e3, 5)
        y = slope * x + intercept
        axs[0].plot(x, y, color=COLOR_V if i < NUM_LINES else COLOR_H, linewidth=LINEWIDTH)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Image Domain", fontsize=BIGSIZE, pad=10)

    axs[1].set_xlim(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
    axs[1].set_ylim(0, 1800)
    for i, (rho, theta) in enumerate(rho_theta):
        if i == NUM_LINES - 2:
            axs[1].plot((theta, theta), (0, rho), linestyle="--", color=GREYCOLOR, linewidth=LINEWIDTH)
            axs[1].plot((-10, theta), (rho, rho), linestyle="--", color=GREYCOLOR, linewidth=LINEWIDTH)
            axs[1].text(theta + 0.06, rho - 20, r"$(\theta_i, \rho_i)$", fontsize=BIGSIZE, color=GREYCOLOR)
        axs[1].plot(theta, rho, "s", color=COLOR_V if i < NUM_LINES else COLOR_H, markersize=8)
    axs[1].set_xticks(np.linspace(-np.pi / 2, np.pi / 2, 5))
    axs[1].set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"], fontsize=NORMALSIZE)
    axs[1].set_yticks([])
    axs[1].set_title("Angle-Radius Domain", fontsize=BIGSIZE, pad=10)
    axs[1].set_xlabel(r"$\theta$", fontsize=BIGSIZE)
    axs[1].set_ylabel(r"$\rho$", fontsize=BIGSIZE)

    plt.tight_layout()
    os.makedirs("src/report/figures/perspective", exist_ok=True)
    plt.savefig("src/report/figures/perspective/angle_radius_domain.pgf", format="pgf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
