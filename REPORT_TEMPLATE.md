# Mesh Normalization, Quantization, and Error Analysis Report

**Student Name:** Aditya Bhushan 
**Date:** 2025-11-09  
**Assignment:** SeamGPT Data Processing - Mesh Normalization, Quantization, and Error Analysis

---

## Executive Summary

This assignment implements a complete pipeline for 3D mesh processing, focusing on normalization, quantization, and error analysis. Two normalization methods (Min-Max and Unit Sphere) were implemented and compared. The pipeline processes mesh files, normalizes vertex coordinates, quantizes them using 1024 bins, and reconstructs the meshes to analyze reconstruction errors. Results show that both normalization methods achieve similar reconstruction quality with very low error rates (MSE ~4.92e-07, MAE ~6.35e-04) when using 1024 quantization bins.

---

## Introduction

Mesh normalization and quantization are fundamental techniques in 3D mesh processing, particularly important for:
- **Data compression**: Reducing storage requirements by quantizing continuous coordinates
- **Machine learning**: Preparing mesh data for neural network training (e.g., SeamGPT)
- **Standardization**: Normalizing meshes to a common coordinate space for comparison
- **Efficient encoding**: Converting floating-point coordinates to discrete integer values

This project implements a complete pipeline that normalizes mesh vertices, quantizes them, and then reconstructs the original mesh to measure the quality loss introduced by quantization.

---

## Methodology

### Normalization Methods

1. **Min-Max Normalization**
   - **Process**: Maps each coordinate axis independently to the [0, 1] range using the formula: `normalized = (vertex - min) / (max - min)`
   - **Advantages**: 
     - Simple and computationally efficient
     - Preserves the aspect ratio of the mesh
     - Direct mapping preserves relative positions
   - **Limitations**: 
     - Sensitive to outliers
     - Does not preserve geometric relationships if axes have different scales
     - May distort meshes with non-uniform bounding boxes

2. **Unit Sphere Normalization**
   - **Process**: 
     1. Centers the mesh by subtracting the centroid
     2. Scales to fit inside a unit sphere (radius = 0.5)
     3. Maps from [-0.5, 0.5] to [0, 1] by adding 0.5
   - **Advantages**: 
     - Preserves geometric relationships and proportions
     - Rotation-invariant (centers mesh at origin)
     - Better for meshes with varying scales
   - **Limitations**: 
     - More computationally expensive (requires distance calculations)
     - May compress meshes with large aspect ratios

### Quantization

- **Process**: The normalized coordinates (in [0, 1]) are quantized by:
  1. Multiplying by (bins - 1) to map to [0, bins-1]
  2. Rounding to nearest integer
  3. Clipping to valid range [0, bins-1]
  
- **Impact of Bin Size on Reconstruction Quality**:
  - **Higher bin count (e.g., 1024)**: 
    - Provides finer resolution and lower quantization error
    - Results in smaller MSE/MAE values
    - Requires more storage (10 bits per coordinate)
    - In our tests with 1024 bins: MSE ≈ 4.92e-07, MAE ≈ 6.35e-04
  
  - **Lower bin count (e.g., 256)**: 
    - Coarser resolution, higher quantization error
    - More compression but visible quality loss
    - Requires less storage (8 bits per coordinate)
  
  - **Trade-off**: There's a direct trade-off between storage efficiency and reconstruction quality. For most applications, 1024 bins (10-bit quantization) provides an excellent balance, achieving sub-millimeter accuracy for typical mesh scales.

---

## Results

### Mesh Statistics

**Test Mesh: test_sphere**
- **Number of vertices**: 162
- **Coordinate ranges**: All axes range from -1.0 to 1.0
- **Centroid**: Approximately at origin (0, 0, 0) - indicating a well-centered mesh
- **Standard deviation**: 0.577 for all axes - indicating uniform distribution

The mesh statistics show a symmetric, well-distributed sphere mesh, which is ideal for comparing normalization methods as both should perform similarly on such a shape.

### Error Analysis

**Min-Max Normalization Results:**
- **Global MSE**: 4.919456e-07
- **Global MAE**: 6.353029e-04
- **Per-axis errors**: Uniform across all axes (X, Y, Z), indicating balanced quantization

**Unit Sphere Normalization Results:**
- **Global MSE**: 4.919460e-07
- **Global MAE**: 6.353032e-04
- **Per-axis errors**: Nearly identical to Min-Max, with uniform distribution

**Key Findings:**
- Both methods achieve extremely low reconstruction errors
- Errors are uniformly distributed across all axes
- The difference between methods is negligible (MSE difference: ~4e-12)
- MAE values indicate average vertex displacement of ~0.0006 units, which is excellent for 1024-bin quantization

### Comparison of Methods

**Similarities:**
- Both methods achieve nearly identical reconstruction quality
- Error metrics are virtually indistinguishable
- Both preserve mesh topology and general shape

**Differences:**
- **Min-Max**: Slightly simpler computation, preserves bounding box aspect ratio
- **Unit Sphere**: Better for meshes that need rotation-invariant representation, centers mesh at origin

**For this test case (sphere)**: Both methods perform equally well because the mesh is already centered and symmetric. The choice between methods would matter more for:
- Meshes with large aspect ratios
- Meshes that are not centered
- Applications requiring rotation-invariant representations

---

## Discussion

### Key Observations

1. **Quantization Quality**: With 1024 bins, both normalization methods achieve excellent reconstruction quality with errors in the order of 10^-7 (MSE) and 10^-4 (MAE), demonstrating that 10-bit quantization is sufficient for high-quality mesh representation.

2. **Method Equivalence**: For symmetric, well-centered meshes like spheres, both normalization methods produce nearly identical results. The choice of method becomes more important for asymmetric or off-center meshes.

3. **Uniform Error Distribution**: The per-axis errors are uniform across X, Y, and Z axes, indicating that the quantization process treats all dimensions equally and does not introduce directional bias.

### Limitations

1. **Fixed Bin Size**: The current implementation uses a fixed bin size (1024). Adaptive quantization based on local vertex density could potentially improve quality.

2. **No Lossless Option**: The pipeline always quantizes, even when lossless compression might be desired for certain applications.

3. **Single Mesh Processing**: The pipeline processes meshes independently. Batch processing with shared normalization parameters could be useful for multi-mesh datasets.

4. **Error Metrics**: Only MSE and MAE are computed. Additional metrics like Hausdorff distance or visual quality metrics could provide more comprehensive evaluation.

### Future Work

1. **Adaptive Quantization**: Implement variable bin sizes based on local vertex density or curvature to optimize quality vs. compression trade-offs.

2. **Seam Tokenization**: As mentioned in the bonus requirements, implement UV seam representation as discrete tokens for encoding/decoding demonstrations.

3. **Multi-resolution Analysis**: Compare reconstruction quality across different bin sizes (256, 512, 1024, 2048) to establish quality vs. storage trade-off curves.

4. **Visual Quality Metrics**: Add perceptual quality metrics beyond numerical error measures.

5. **Batch Processing**: Extend to process multiple meshes with shared normalization parameters for consistent representation.

---

## Conclusion

This project successfully implements a complete mesh normalization, quantization, and error analysis pipeline. Both Min-Max and Unit Sphere normalization methods achieve excellent reconstruction quality with 1024-bin quantization, producing errors on the order of 10^-7 (MSE) and 10^-4 (MAE). 

The results demonstrate that:
- 10-bit quantization (1024 bins) provides sufficient precision for high-quality mesh reconstruction
- Both normalization methods are effective, with choice depending on mesh characteristics and application requirements
- The pipeline successfully processes meshes, generates comprehensive metrics, and produces visualizations for analysis

The implementation is complete, well-documented, and ready for use in mesh processing workflows, particularly for applications requiring quantized mesh representations such as SeamGPT data processing.

---

## References

- Trimesh Library: https://github.com/mikedh/trimesh
- NumPy Documentation: https://numpy.org/doc/
- Mesh Processing Fundamentals: Standard techniques in 3D computer graphics

---

**Note:** This template has been filled with actual results and analysis. The automated report (REPORT.md) contains all computed metrics and statistics in tabular format.

