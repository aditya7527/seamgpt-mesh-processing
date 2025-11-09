# Outputs Directory

This directory contains the generated outputs from the mesh processing pipeline.

## Structure

Each pipeline run creates a timestamped directory:
```
outputs/
└── report_YYYYMMDD_HHMMSS/
    ├── images/          # Visualization plots
    ├── meshes/          # Processed mesh files
    ├── metrics.json     # Computed metrics
    └── REPORT.md        # Auto-generated report
```

## Contents

### Images (`images/`)
- `{mesh_name}_scatter_plots.png` - 2D vertex projections (XY, XZ, YZ)
- `{mesh_name}_error_analysis.png` - Per-axis MSE/MAE bar charts
- `{mesh_name}_mse_comparison.png` - Global MSE comparison between methods

### Meshes (`meshes/`)
- `{mesh_name}_normalized_minmax.obj` - Min-Max normalized mesh
- `{mesh_name}_normalized_sphere.obj` - Unit Sphere normalized mesh
- `{mesh_name}_quantized_minmax.obj` - Quantized mesh (Min-Max)
- `{mesh_name}_quantized_sphere.obj` - Quantized mesh (Unit Sphere)
- `{mesh_name}_reconstructed_minmax.obj` - Reconstructed mesh (Min-Max)
- `{mesh_name}_reconstructed_sphere.obj` - Reconstructed mesh (Unit Sphere)

### Data Files
- `metrics.json` - All computed metrics in JSON format
- `REPORT.md` - Auto-generated Markdown report with statistics and analysis

## Note

Output files are excluded from Git (via `.gitignore`) because:
- They are generated files that can be recreated by running the pipeline
- They can be large (especially mesh files)
- Each run creates new timestamped directories

To generate outputs, run:
```bash
python seamgpt_mesh_pipeline.py --input_dir meshes --output_dir outputs --bins 1024
```

