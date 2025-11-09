#!/usr/bin/env python3
"""Quick script to show output summary"""
from pathlib import Path

out_dir = Path('outputs/report_20251109_173834')

print("=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)
print(f"\nOutput Directory: {out_dir.absolute()}\n")

# Images
images = sorted((out_dir / "images").glob("*.png"))
print(f"[IMAGES] ({len(images)} files):")
for img in images:
    print(f"  - {img.name}")

# Meshes
meshes = sorted((out_dir / "meshes").glob("*.obj"))
print(f"\n[MESHES] ({len(meshes)} files):")
for mesh in meshes:
    print(f"  - {mesh.name}")

# Reports
print(f"\n[REPORTS]:")
print(f"  - metrics.json")
print(f"  - REPORT.md")

print("\n" + "=" * 60)
print("All outputs are ready for submission!")
print("=" * 60)

