# SeamGPT Mesh Processing Pipeline

Mesh Normalization, Quantization, and Error Analysis

## Overview

This project implements a complete pipeline for processing 3D meshes:
- Loading and inspecting mesh files
- Normalization (Min-Max and Unit Sphere methods)
- Quantization with configurable bins
- Reconstruction and error analysis
- Automated report generation

## Requirements

```bash
pip install numpy trimesh matplotlib
```

Optional (for enhanced visualization):
```bash
pip install open3d
```

## Project Structure

```
seamgpt_mesh_assignment/
│
├── seamgpt_mesh_pipeline.py         # Main script (CLI)
├── seamgpt_mesh_pipeline.ipynb      # Notebook runner
├── README.md                        # This file
├── REPORT_TEMPLATE.md               # Editable report template
├── outputs/
│   └── report_<timestamp>/
│       ├── images/                  # All plots
│       ├── meshes/                  # Normalized / quantized / reconstructed
│       ├── metrics.json
│       └── REPORT.md
└── meshes/                          # Input .obj files (create this folder)
```

## Usage

### Command Line Interface

```bash
python seamgpt_mesh_pipeline.py \
  --input_dir meshes \
  --output_dir outputs \
  --bins 1024
```

**Arguments:**
- `--input_dir`: Directory containing `.obj` files (default: `meshes`)
- `--output_dir`: Output directory for results (default: `outputs`)
- `--bins`: Number of quantization bins (default: `1024`)

### Jupyter Notebook

Open `seamgpt_mesh_pipeline.ipynb` and run the cells interactively.

## Features

### Task 1: Load & Inspect Mesh
- Loads all `.obj` files from the input directory
- Extracts vertices as NumPy arrays
- Computes and displays statistics (min, max, mean, std per axis)
- Generates 2D scatter plots (XY, XZ, YZ projections)

### Task 2: Normalize & Quantize
- **Min-Max Normalization**: Maps coordinates to [0, 1] range
- **Unit Sphere Normalization**: Centers mesh, scales to unit sphere, then maps to [0, 1]
- Quantizes normalized coordinates with specified number of bins
- Saves normalized and quantized meshes

### Task 3: Dequantize, Denormalize & Error Analysis
- Reconstructs meshes by reversing quantization and normalization
- Computes MSE and MAE (global and per-axis)
- Generates error visualization plots:
  - Per-axis MSE/MAE bar charts
  - Global MSE comparison between methods
- Saves reconstructed meshes

### Metrics & Report
- Saves all metrics to `metrics.json`
- Auto-generates `REPORT.md` with:
  - Statistics summary
  - Error values and comparisons
  - Observations and analysis

## Output Files

For each processed mesh, the pipeline generates:

1. **Normalized meshes:**
   - `{mesh_name}_normalized_minmax.obj`
   - `{mesh_name}_normalized_sphere.obj`

2. **Quantized meshes:**
   - `{mesh_name}_quantized_minmax.obj`
   - `{mesh_name}_quantized_sphere.obj`

3. **Reconstructed meshes:**
   - `{mesh_name}_reconstructed_minmax.obj`
   - `{mesh_name}_reconstructed_sphere.obj`

4. **Visualizations:**
   - `{mesh_name}_scatter_plots.png` - Original vertex projections
   - `{mesh_name}_error_analysis.png` - Per-axis error charts
   - `{mesh_name}_mse_comparison.png` - Global MSE comparison

5. **Data:**
   - `metrics.json` - All computed metrics
   - `REPORT.md` - Auto-generated report

## Example

```bash
# Create meshes directory and add your .obj files
mkdir meshes
# Copy your .obj files to meshes/

# Run the pipeline
python seamgpt_mesh_pipeline.py --input_dir meshes --bins 1024

# Results will be in outputs/report_YYYYMMDD_HHMMSS/
```

## Notes

- The pipeline processes all `.obj` files found in the input directory
- Each run creates a timestamped output directory to avoid overwriting previous results
- All plots are saved as PNG files with 150 DPI resolution
- The report can be converted to PDF using tools like Pandoc or Markdown viewers

## License

This project is for educational/assignment purposes.

