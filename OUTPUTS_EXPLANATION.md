# Outputs Directory Explanation

## About the Outputs Folder

The `outputs/` directory contains all generated results from running the mesh processing pipeline. These files are **excluded from Git** for the following reasons:

### Why Outputs Are Excluded

1. **Generated Files**: All outputs can be recreated by running the pipeline
2. **File Size**: Mesh files and images can be large
3. **Version Control**: Each run creates timestamped directories, leading to many files
4. **Best Practice**: Only source code and documentation should be in version control

### What's in the Outputs

Based on your latest run (`report_20251109_175218`), the outputs include:

#### Images (3 files)
- `test_sphere_scatter_plots.png` - 2D vertex projections
- `test_sphere_error_analysis.png` - Error analysis charts
- `test_sphere_mse_comparison.png` - MSE comparison

#### Meshes (6 files)
- Normalized meshes (MinMax & Unit Sphere)
- Quantized meshes (MinMax & Unit Sphere)
- Reconstructed meshes (MinMax & Unit Sphere)

#### Reports
- `metrics.json` - All computed metrics
- `REPORT.md` - Auto-generated report

### How to Generate Outputs

Simply run the pipeline:
```bash
python seamgpt_mesh_pipeline.py --input_dir meshes --output_dir outputs --bins 1024
```

Outputs will be created in: `outputs/report_YYYYMMDD_HHMMSS/`

### Viewing Your Outputs

Your outputs are located at:
```
C:\Users\ASUS\Desktop\Assign\outputs\report_20251109_175218\
```

You can:
- View images in the `images/` folder
- Open mesh files in 3D viewers
- Read the `REPORT.md` for analysis
- Check `metrics.json` for numerical data

### Git Status

✅ **Tracked in Git:**
- `outputs/.gitkeep` - Ensures folder exists
- `outputs/README.md` - Documentation

❌ **Excluded from Git:**
- All generated files (meshes, images, reports)
- All timestamped report directories

This keeps your repository clean while preserving the folder structure!

