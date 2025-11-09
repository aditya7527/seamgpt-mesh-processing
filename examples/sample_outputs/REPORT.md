# Mesh Normalization, Quantization, and Error Analysis Report

**Generated:** 2025-11-09 17:52:20
**Number of bins:** 1024

---

## Mesh: test_sphere

### Statistics

- **Number of vertices:** 162

**Per-axis statistics:**

| Axis | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| X | -1.000000 | 1.000000 | 0.000000 | 0.577350 |
| Y | -1.000000 | 1.000000 | -0.000000 | 0.577350 |
| Z | -1.000000 | 1.000000 | 0.000000 | 0.577350 |

### MinMax Normalization Results

**Normalization Parameters:**
- Min: [-1.0, -1.0, -1.0]
- Max: [1.0, 1.0, 1.0]

**Error Metrics:**

| Metric | Global | X-axis | Y-axis | Z-axis |
|--------|--------|--------|--------|--------|
| MSE | 4.919456e-07 | 4.919456e-07 | 4.919456e-07 | 4.919456e-07 |
| MAE | 6.353029e-04 | 6.353029e-04 | 6.353029e-04 | 6.353029e-04 |

### Unit Sphere Normalization Results

**Normalization Parameters:**
- Centroid: [3.974872557299943e-17, -4.454598555594764e-18, 1.1650488530017074e-17]
- Scale: 0.500000
- Max Distance: 1.000000

**Error Metrics:**

| Metric | Global | X-axis | Y-axis | Z-axis |
|--------|--------|--------|--------|--------|
| MSE | 4.919460e-07 | 4.919460e-07 | 4.919460e-07 | 4.919460e-07 |
| MAE | 6.353032e-04 | 6.353032e-04 | 6.353032e-04 | 6.353032e-04 |

### Comparison

**Global MSE:**
- MinMax: 4.919456e-07
- Unit Sphere: 4.919460e-07

**Global MAE:**
- MinMax: 6.353029e-04
- Unit Sphere: 6.353032e-04

### Visualizations

- Scatter plots: `test_sphere_scatter_plots.png`
- Error analysis: `test_sphere_error_analysis.png`
- MSE comparison: `test_sphere_mse_comparison.png`

---

## Summary

This report summarizes the mesh normalization, quantization, and error analysis results.
Two normalization methods were compared:

1. **MinMax Normalization:** Maps coordinates directly to [0, 1] based on min/max values.
2. **Unit Sphere Normalization:** Centers the mesh, scales to fit a unit sphere, then maps to [0, 1].

Both methods were quantized using 1024 bins, then reconstructed to compute error metrics.

### Observations

- The error metrics show the reconstruction quality after quantization.
- Lower MSE/MAE values indicate better preservation of the original mesh geometry.
- The choice of normalization method may affect error distribution across axes.
