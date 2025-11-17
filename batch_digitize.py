#!/usr/bin/env python3
"""
Batch digitize multiple images in a folder.
Each image is digitized separately and saved with its own JSON and combined plot.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch digitize all images in a folder.")
    parser.add_argument("folder", type=str, help="Path to folder containing images")
    parser.add_argument("--pattern", type=str, default="*.png", help="File pattern to match (default: *.png)")
    parser.add_argument("--force", action="store_true", help="Re-digitize even if output already exists")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory")
        return

    # Find all matching images
    images = sorted(folder.glob(args.pattern))
    if not images:
        print(f"No images found matching pattern '{args.pattern}' in '{folder}'")
        return

    # Check which images already have output
    output_dir = Path("./digitized_output")
    skipped = []
    to_process = []

    for img in images:
        base_name = img.stem
        json_file = output_dir / f"{base_name}_digitized.json"
        png_file = output_dir / f"{base_name}_digitized.png"

        if args.force or not (json_file.exists() and png_file.exists()):
            to_process.append(img)
        else:
            skipped.append(img)

    print(f"Found {len(images)} images matching pattern '{args.pattern}'")
    if skipped:
        print(f"\nAlready digitized ({len(skipped)}):")
        for img in skipped:
            print(f"  ✓ {img.name}")
    if to_process:
        print(f"\nTo be digitized ({len(to_process)}):")
        for img in to_process:
            print(f"  - {img.name}")
    else:
        print("\nAll images already digitized. Use --force to re-digitize.")
        return

    print("\n" + "="*60)
    proceed = input("Proceed with digitization? [y/n]: ").strip().lower()
    if proceed not in ("y", "yes"):
        print("Aborted.")
        return

    images = to_process  # Only process remaining images

    # Process each image
    for i, img_path in enumerate(images, 1):
        print("\n" + "="*60)
        print(f"Processing image {i}/{len(images)}: {img_path.name}")
        print("="*60)

        # Run the digitize script
        cmd = ["python", "digitize_and_replot.py", str(img_path)]
        if args.force:
            cmd.append("--force")

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Successfully digitized {img_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to digitize {img_path.name}: {e}")
            proceed = input("Continue with next image? [y/n]: ").strip().lower()
            if proceed not in ("y", "yes"):
                print("Stopping batch processing.")
                break
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break

    print("\n" + "="*60)
    print("Batch processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
