
import json
import os
import argparse
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from datetime import datetime

# ---------- CONFIG ----------
IMAGE_PATH = None  # Will be set via command-line argument

# Beta ticks visible on the x-axes in the paper
BETA_TICKS = [1.0, 1.333, 1.666, 2.0]

# Number of panels to digitize
NUM_PANELS = 1  # Default to 1, can be changed via command line

# Epsilon curves to capture per panel (order matters for your workflow)
EPS_LIST = [0.5, 1.0, 2.0, 3.0, 4.0]

# Where to write outputs (will be set in main based on image filename)
OUT_DIR = None
OUT_JSON = None

# ---------- HELPERS ----------
@dataclass
class AxisCal:
    x0: float  # pixel
    y0: float  # pixel
    x1: float  # pixel
    y1: float  # pixel
    data_x0: float
    data_x1: float
    data_y0: float
    data_y1: float

    def pix_to_data(self, xpix, ypix):
        # Linear mapping from pixel to data based on two corners (BL, TR)
        x = self.data_x0 + (xpix - self.x0) * (self.data_x1 - self.data_x0) / (self.x1 - self.x0)
        # Note: pixel y increases downward, data y upward
        y = self.data_y0 + (self.y0 - ypix) * (self.data_y1 - self.data_y0) / (self.y0 - self.y1)
        return x, y

def ginput_points(n, prompt):
    print(prompt)
    pts = []
    for i in range(n):
        print(f"  Waiting for point {i+1}/{n}...")
        pt = plt.ginput(1, timeout=-1)
        if len(pt) == 0:
            raise RuntimeError(f"Expected {n} points, got {len(pts)}")
        pts.append(pt[0])
        print(f"  ✓ Recorded point {i+1}/{n}: ({pt[0][0]:.1f}, {pt[0][1]:.1f})")
    return pts

def ask_yes_no(prompt):
    while True:
        s = input(prompt + " [y/n]: ").strip().lower()
        if s in ("y","yes"):
            return True
        if s in ("n","no"):
            return False

# ---------- MAIN INTERACTIVE FLOW ----------
def digitize_subfigure(fig, ax, panel_name, y_min, y_max):
    """Digitize a single subfigure from the displayed image."""
    ax.set_title(f"Digitizing: {panel_name}\nClick BOTTOM-LEFT, then TOP-RIGHT corners of the plotting area.")
    plt.draw()

    print(f"\n=== Digitizing: {panel_name} ===")
    print(f"Y-axis range: [{y_min}, {y_max}]")

    # Calibrate axes corners
    bl, tr = ginput_points(2, "Click BOTTOM-LEFT, then TOP-RIGHT corners of the plotting area.")
    (x0, y0), (x1, y1) = bl, tr

    # Build calibration mapping (x maps 1.0 -> 4.0, y maps y_min -> y_max)
    cal = AxisCal(x0=x0, y0=y0, x1=x1, y1=y1,
                  data_x0=BETA_TICKS[0], data_x1=BETA_TICKS[-1],
                  data_y0=y_min, data_y1=y_max)

    panel_data = {}

    for eps in EPS_LIST:
        print(f"\n-- Epsilon {eps} --")
        print("You will click points in order of beta ticks:")
        print(BETA_TICKS)
        print("If a value is missing, you can click the same y twice (or approximate).")

        pts = ginput_points(len(BETA_TICKS), f"Click {len(BETA_TICKS)} points along the curve for eps={eps}.")
        yvals = []
        for i, ((xpix, ypix), beta) in enumerate(zip(pts, BETA_TICKS)):
            x_data, y_data = cal.pix_to_data(xpix, ypix)
            # We lock x to the closest beta tick; store only y
            yvals.append(float(y_data))
            print(f"  Point {i+1}/{len(BETA_TICKS)}: beta={beta}, y={y_data:.2f}")

        panel_data[str(eps)] = yvals
        print(f"Captured eps={eps}: {np.round(yvals, 2).tolist()}")

    # Overlay preview dots (optional)
    for eps, yvals in panel_data.items():
        xs = BETA_TICKS
        # Map data to pixel for preview
        pix = [ (cal.x0 + (x-cal.data_x0)*(cal.x1-cal.x0)/(cal.data_x1-cal.data_x0),
                 cal.y0 - (y-cal.data_y0)*(cal.y0-cal.y1)/(cal.data_y1-cal.data_y0)) for x,y in zip(xs, yvals)]
        xs_pix, ys_pix = zip(*pix)
        ax.plot(xs_pix, ys_pix, marker='o', linestyle='-')

    plt.draw()

    return panel_data

def replot_combined_figure(beta, panels, base_name):
    """Create a 2x2 subplot figure with all 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (panel_name, d) in enumerate(panels.items()):
        ax = axes[idx]
        # Plot each eps' y-curve
        for eps_str, yvals in d.items():
            ax.plot(beta, yvals, marker='o', label=f"\u03B5 = {eps_str}")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Value")
        ax.set_title(panel_name)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save combined figure with base name
    out_path = os.path.join(OUT_DIR, f"{base_name}_digitized.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved combined figure: {out_path}")
    plt.show()

def main():
    global OUT_DIR, OUT_JSON

    parser = argparse.ArgumentParser(description="Digitize and replot data from 4 subfigures in a single image.")
    parser.add_argument("image_path", type=str, help="Path to the image with 4 subfigures")
    parser.add_argument("--force", action="store_true", help="Force re-digitization even if output exists")
    args = parser.parse_args()

    # Verify image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        return

    # Setup output directory and files based on image name
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    OUT_DIR = f"./digitized_output"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_JSON = os.path.join(OUT_DIR, f"{base_name}_digitized.json")
    OUT_PNG = os.path.join(OUT_DIR, f"{base_name}_digitized.png")

    # Check if output already exists
    if not args.force and os.path.exists(OUT_JSON) and os.path.exists(OUT_PNG):
        print(f"Output already exists for '{base_name}':")
        print(f"  - {OUT_JSON}")
        print(f"  - {OUT_PNG}")
        print("Skipping. Use --force to re-digitize.")
        return

    # Panel names in order: top-left, top-right, bottom-left, bottom-right
    PANEL_NAMES = [
        "LSTM on IMDB",
        "ScatterNet on SVHN",
        "ScatterNet on CIFAR-10",
        "FCN on Adult"
    ]

    # Show the image and keep it displayed
    print("\n" + "=" * 60)
    print("Displaying image for reference...")
    print("The image will stay open while you provide input.")
    print("=" * 60)

    img = plt.imread(args.image_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Reference: {base_name}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)  # Give time for window to appear

    # Collect metadata for all 4 subfigures
    print("\n" + "=" * 60)
    print("Please provide y-axis ranges for all 4 subfigures")
    print("=" * 60)

    panel_configs = []
    for i, panel_name in enumerate(PANEL_NAMES):
        print(f"\n--- Subfigure {i+1}/4: {panel_name} ---")
        y_min = float(input(f"  Enter y-axis minimum for '{panel_name}': "))
        y_max = float(input(f"  Enter y-axis maximum for '{panel_name}': "))
        panel_configs.append((panel_name, y_min, y_max))

    # Now digitize all 4 subfigures from the same image
    all_panels = {}
    for i, (panel_name, y_min, y_max) in enumerate(panel_configs, 1):
        print(f"\n{'='*60}")
        print(f"Digitizing subfigure {i}/4: {panel_name}")
        print(f"{'='*60}")
        panel_data = digitize_subfigure(fig, ax, panel_name, y_min, y_max)
        all_panels[panel_name] = panel_data

    # Close the image after all digitization is done
    plt.close(fig)

    # Save JSON
    output_data = {"beta": BETA_TICKS, "panels": all_panels}
    with open(OUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved digitized data to {OUT_JSON}")
    print(f"{'='*60}")

    # Create combined 2x2 plot
    replot_combined_figure(BETA_TICKS, all_panels, base_name)

    print("\nSummary JSON:\n", json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()
