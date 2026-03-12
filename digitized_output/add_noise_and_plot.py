#!/usr/bin/env python3
"""
Add noise/std deviations to digitized data and create plots with error bars.
Reads all JSON files in digitized_output/, adds random std devs, and creates plots.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
MEAN_STD_DEV = 0.25  # Std dev of the Gaussian from which std devs are drawn
OUTPUT_DIR = "./images_std_dev"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

def add_noise_to_data(data):
    """Add random standard deviations to the data."""
    # Generate random std devs for each point independently
    data_with_std = {
        "beta": data["beta"],
        "panels": {}
    }

    for panel_name, eps_dict in data["panels"].items():
        data_with_std["panels"][panel_name] = {}

        for eps, y_values in eps_dict.items():
            # Sample a different std dev for EACH point from N(0, 0.25), take absolute value
            std_devs = [np.abs(np.random.normal(0, MEAN_STD_DEV)) for _ in range(len(y_values))]

            data_with_std["panels"][panel_name][eps] = {
                "mean": y_values,
                "std": std_devs
            }

    return data_with_std

def get_suptitle(filename):
    """Generate suptitle based on filename."""
    if "lbeta.png" in filename or filename == "lbeta_with_std.png":
        return None

    # Extract the base name without _with_std
    base = filename.replace("_with_std.png", "")

    if "sigmas" in base:
        # Extract number after "sigmas_"
        parts = base.split("sigmas_")
        if len(parts) > 1:
            value = parts[1].replace(".png", "")
            return f"Noise multiplier fixed to {value}"

    elif "LR" in base:
        # Extract number after "LR_"
        parts = base.split("LR_")
        if len(parts) > 1:
            value = parts[1].replace(".png", "")
            return f"Learning rate fixed to {value}"

    elif "max_grad_norm" in base:
        # Extract number after "max_grad_norm_"
        parts = base.split("max_grad_norm_")
        if len(parts) > 1:
            value = parts[1].replace(".png", "")
            return f"Clipping norm fixed to {value}"

    elif "batch_sizes" in base:
        # Extract number after "batch_sizes_"
        parts = base.split("batch_sizes_")
        if len(parts) > 1:
            value = parts[1].replace(".png", "")
            return f"Batch size fixed to {value}"

    return None

def plot_with_error_bars(beta, panels, output_path):
    """Create a 2x2 subplot figure with all 4 panels including error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    panel_names = list(panels.keys())

    for idx, panel_name in enumerate(panel_names):
        if idx >= len(axes):
            break

        ax = axes[idx]
        panel_data = panels[panel_name]

        # Collect all y values (with error bars) to determine ylim
        all_y_values = []

        # Plot each epsilon curve with error bars
        for eps_str, values in panel_data.items():
            y_mean = values["mean"]
            y_std = values["std"]

            # Collect min and max including error bars
            all_y_values.extend(np.array(y_mean) - np.array(y_std))
            all_y_values.extend(np.array(y_mean) + np.array(y_std))

            # Plot line with markers
            line = ax.plot(beta, y_mean, marker='o', label=f"ε = {eps_str}", linewidth=2)
            color = line[0].get_color()

            # Add filled error region
            ax.fill_between(beta,
                           np.array(y_mean) - np.array(y_std),
                           np.array(y_mean) + np.array(y_std),
                           alpha=0.2, color=color)

        # Set y-limits with buffer
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min

        if y_range < 4:
            # If range is less than 4, use fixed buffer of 3.5
            ylim_min = y_min - 3.5
            ylim_max = y_max + 3.5
        else:
            # Otherwise use 10% buffer
            buffer = 0.1 * y_range
            ylim_min = y_min - buffer
            ylim_max = y_max + buffer

        ax.set_ylim(ylim_min, ylim_max)

        # Only show x-axis label on bottom row (indices 2, 3)
        if idx >= 2:
            ax.set_xlabel(r"$\beta$", fontsize=12)

        # Only show y-axis label on left column (indices 0, 2)
        if idx % 2 == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=12)

        ax.set_title(panel_name, fontsize=13, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved PNG: {output_path}")

    # Also save as PDF
    pdf_path = str(output_path).replace("/plots/", "/pdfs/").replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved PDF: {pdf_path}")

    plt.close(fig)

def main():
    # Create output directories
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    # Find all JSON files in current directory (digitized_output)
    json_files = sorted(Path(".").glob("*_digitized.json"))

    if not json_files:
        print("No digitized JSON files found in digitized_output/")
        return

    print(f"Found {len(json_files)} JSON files to process:")
    for json_file in json_files:
        print(f"  - {json_file.name}")

    print("\n" + "="*60)

    # Process each JSON file
    for json_file in json_files:
        base_name = json_file.stem.replace("_digitized", "")
        print(f"\nProcessing: {json_file.name}")

        # Load original data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Add noise/std deviations
        data_with_std = add_noise_to_data(data)

        # Save new JSON with std devs to json/ subdirectory
        output_json = Path(JSON_DIR) / f"{base_name}_with_std.json"
        with open(output_json, 'w') as f:
            json.dump(data_with_std, f, indent=2)
        print(f"  Saved JSON: {output_json}")

        # Create plot with error bars in plots/ subdirectory
        output_png = Path(PLOTS_DIR) / f"{base_name}_with_std.png"
        plot_with_error_bars(data["beta"], data_with_std["panels"], output_png)

    print("\n" + "="*60)
    print("All files processed successfully!")
    print(f"JSONs saved to: {JSON_DIR}/")
    print(f"PNGs saved to: {PLOTS_DIR}/")
    print(f"PDFs saved to: {PDF_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
