#!/usr/bin/env python3
"""
SeamGPT Mesh Processing Pipeline
Mesh Normalization, Quantization, and Error Analysis
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh


class MeshProcessor:
    """Main class for processing meshes: normalization, quantization, and error analysis."""
    
    def __init__(self, input_dir: str, output_dir: str, bins: int = 1024):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.bins = bins
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = self.output_dir / f"report_{self.timestamp}"
        self.images_dir = self.run_output_dir / "images"
        self.meshes_dir = self.run_output_dir / "meshes"
        
        # Create output directories
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
    
    def load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        """Load a mesh from file."""
        try:
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            mesh = trimesh.load(str(mesh_path))
            if mesh is None:
                raise ValueError(f"Failed to load mesh from {mesh_path}")
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                meshes = list(mesh.geometry.values())
                if not meshes:
                    raise ValueError(f"Scene has no meshes: {mesh_path}")
                mesh = meshes[0]
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise ValueError(f"Mesh has no vertices: {mesh_path}")
            return mesh
        except Exception as e:
            print(f"Error loading {mesh_path}: {e}")
            raise
    
    def extract_vertices(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Extract vertices as NumPy array."""
        vertices = mesh.vertices.copy()
        if len(vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if vertices.shape[1] != 3:
            raise ValueError(f"Expected 3D vertices, got shape {vertices.shape}")
        return vertices
    
    def compute_stats(self, vertices: np.ndarray) -> Dict:
        """Compute statistics for vertices."""
        stats = {
            'num_vertices': len(vertices),
            'min': vertices.min(axis=0).tolist(),
            'max': vertices.max(axis=0).tolist(),
            'mean': vertices.mean(axis=0).tolist(),
            'std': vertices.std(axis=0).tolist(),
        }
        return stats
    
    def print_stats(self, mesh_name: str, stats: Dict):
        """Print statistics to console."""
        print(f"\n{'='*60}")
        print(f"Mesh: {mesh_name}")
        print(f"{'='*60}")
        print(f"Number of vertices: {stats['num_vertices']}")
        print(f"\nPer-axis statistics:")
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            print(f"  {axis}:")
            print(f"    Min:  {stats['min'][i]:.6f}")
            print(f"    Max:  {stats['max'][i]:.6f}")
            print(f"    Mean: {stats['mean'][i]:.6f}")
            print(f"    Std:  {stats['std'][i]:.6f}")
    
    def create_scatter_plots(self, vertices: np.ndarray, mesh_name: str):
        """Create 2D scatter plots (XY, XZ, YZ)."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            projections = [
                (0, 1, 'XY'),
                (0, 2, 'XZ'),
                (1, 2, 'YZ')
            ]
            
            for idx, (ax_idx1, ax_idx2, label) in enumerate(projections):
                axes[idx].scatter(vertices[:, ax_idx1], vertices[:, ax_idx2], 
                                 s=1, alpha=0.5)
                axes[idx].set_xlabel(f'Axis {ax_idx1}')
                axes[idx].set_ylabel(f'Axis {ax_idx2}')
                axes[idx].set_title(f'{label} Projection')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.images_dir / f"{mesh_name}_scatter_plots.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved scatter plots: {plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping scatter plots")
        except Exception as e:
            print(f"Warning: Error creating scatter plots for {mesh_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def normalize_minmax(self, vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Min-Max normalization: map coordinates to [0, 1]."""
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_range = v_max - v_min
        
        # Avoid division by zero
        v_range = np.where(v_range == 0, 1, v_range)
        
        normalized = (vertices - v_min) / v_range
        
        params = {
            'min': v_min.tolist(),
            'max': v_max.tolist(),
            'range': v_range.tolist()
        }
        
        return normalized, params
    
    def normalize_unit_sphere(self, vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Unit Sphere normalization: center mesh, scale to fit unit sphere, then map to [0, 1]."""
        # Center the mesh
        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        
        # Scale to fit inside unit sphere (radius = 0.5)
        max_dist = np.linalg.norm(centered, axis=1).max()
        if max_dist > 0:
            scale = 0.5 / max_dist
        else:
            scale = 1.0
        
        scaled = centered * scale
        
        # Map from [-0.5, 0.5] to [0, 1]
        normalized = scaled + 0.5
        
        params = {
            'centroid': centroid.tolist(),
            'scale': float(scale),
            'max_distance': float(max_dist)
        }
        
        return normalized, params
    
    def quantize(self, normalized_vertices: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quantize normalized coordinates with specified number of bins."""
        # Quantize to bins
        quantized = np.round(normalized_vertices * (self.bins - 1))
        quantized = np.clip(quantized, 0, self.bins - 1).astype(np.int32)
        
        params = {
            'bins': self.bins,
            'min_value': 0,
            'max_value': self.bins - 1
        }
        
        return quantized, params
    
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize quantized coordinates back to [0, 1]."""
        dequantized = quantized.astype(np.float64) / (self.bins - 1)
        return dequantized
    
    def denormalize_minmax(self, normalized: np.ndarray, params: Dict) -> np.ndarray:
        """Reverse Min-Max normalization."""
        v_min = np.array(params['min'])
        v_max = np.array(params['max'])
        v_range = v_max - v_min
        
        denormalized = normalized * v_range + v_min
        return denormalized
    
    def denormalize_unit_sphere(self, normalized: np.ndarray, params: Dict) -> np.ndarray:
        """Reverse Unit Sphere normalization."""
        # Map from [0, 1] back to [-0.5, 0.5]
        scaled = normalized - 0.5
        
        # Reverse scaling
        scale = params['scale']
        centered = scaled / scale
        
        # Reverse centering
        centroid = np.array(params['centroid'])
        denormalized = centered + centroid
        
        return denormalized
    
    def compute_errors(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict:
        """Compute MSE and MAE (global and per-axis)."""
        errors = {}
        
        # Global errors
        mse_global = np.mean((original - reconstructed) ** 2)
        mae_global = np.mean(np.abs(original - reconstructed))
        
        errors['global'] = {
            'MSE': float(mse_global),
            'MAE': float(mae_global)
        }
        
        # Per-axis errors
        mse_per_axis = np.mean((original - reconstructed) ** 2, axis=0)
        mae_per_axis = np.mean(np.abs(original - reconstructed), axis=0)
        
        errors['per_axis'] = {
            'MSE': mse_per_axis.tolist(),
            'MAE': mae_per_axis.tolist()
        }
        
        return errors
    
    def create_error_plots(self, mesh_name: str, errors_minmax: Dict, errors_sphere: Dict):
        """Create error visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Per-axis MSE/MAE bar charts
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes = axes.flatten()
            
            # Per-axis MSE - MinMax
            axes[0].bar(['X', 'Y', 'Z'], errors_minmax['per_axis']['MSE'], 
                       color='skyblue', alpha=0.7)
            axes[0].set_title('Per-Axis MSE - MinMax Normalization')
            axes[0].set_ylabel('MSE')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Per-axis MAE - MinMax
            axes[1].bar(['X', 'Y', 'Z'], errors_minmax['per_axis']['MAE'], 
                       color='lightcoral', alpha=0.7)
            axes[1].set_title('Per-Axis MAE - MinMax Normalization')
            axes[1].set_ylabel('MAE')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Per-axis MSE - Unit Sphere
            axes[2].bar(['X', 'Y', 'Z'], errors_sphere['per_axis']['MSE'], 
                       color='lightgreen', alpha=0.7)
            axes[2].set_title('Per-Axis MSE - Unit Sphere Normalization')
            axes[2].set_ylabel('MSE')
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Per-axis MAE - Unit Sphere
            axes[3].bar(['X', 'Y', 'Z'], errors_sphere['per_axis']['MAE'], 
                       color='plum', alpha=0.7)
            axes[3].set_title('Per-Axis MAE - Unit Sphere Normalization')
            axes[3].set_ylabel('MAE')
            axes[3].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_path1 = self.images_dir / f"{mesh_name}_error_analysis.png"
            plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved error analysis plot: {plot_path1}")
            
            # Global MSE comparison
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            methods = ['MinMax', 'Unit Sphere']
            mse_values = [errors_minmax['global']['MSE'], errors_sphere['global']['MSE']]
            colors = ['skyblue', 'lightgreen']
            
            bars = ax.bar(methods, mse_values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('Global MSE Comparison')
            ax.set_ylabel('MSE')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, mse_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.6e}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plot_path2 = self.images_dir / f"{mesh_name}_mse_comparison.png"
            plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved MSE comparison plot: {plot_path2}")
            
        except ImportError:
            print("Warning: matplotlib not available, skipping error plots")
        except Exception as e:
            print(f"Warning: Error creating plots for {mesh_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def process_mesh(self, mesh_path: Path):
        """Process a single mesh through the complete pipeline."""
        mesh_name = mesh_path.stem
        print(f"\n{'#'*60}")
        print(f"Processing: {mesh_name}")
        print(f"{'#'*60}")
        
        # Load mesh
        mesh = self.load_mesh(mesh_path)
        vertices = self.extract_vertices(mesh)
        
        # Task 1: Inspect mesh
        stats = self.compute_stats(vertices)
        self.print_stats(mesh_name, stats)
        self.create_scatter_plots(vertices, mesh_name)
        
        # Task 2: Normalize & Quantize
        # Method 1: Min-Max
        normalized_minmax, params_minmax = self.normalize_minmax(vertices)
        quantized_minmax, qparams_minmax = self.quantize(normalized_minmax)
        
        # Method 2: Unit Sphere
        normalized_sphere, params_sphere = self.normalize_unit_sphere(vertices)
        quantized_sphere, qparams_sphere = self.quantize(normalized_sphere)
        
        # Save normalized meshes
        mesh_minmax = mesh.copy()
        mesh_minmax.vertices = normalized_minmax
        mesh_minmax_path = self.meshes_dir / f"{mesh_name}_normalized_minmax.obj"
        mesh_minmax.export(str(mesh_minmax_path))
        
        mesh_sphere = mesh.copy()
        mesh_sphere.vertices = normalized_sphere
        mesh_sphere_path = self.meshes_dir / f"{mesh_name}_normalized_sphere.obj"
        mesh_sphere.export(str(mesh_sphere_path))
        
        # Save quantized meshes (dequantize first to save as OBJ)
        dequantized_minmax = self.dequantize(quantized_minmax)
        mesh_quantized_minmax = mesh.copy()
        mesh_quantized_minmax.vertices = dequantized_minmax
        mesh_quantized_minmax_path = self.meshes_dir / f"{mesh_name}_quantized_minmax.obj"
        mesh_quantized_minmax.export(str(mesh_quantized_minmax_path))
        
        dequantized_sphere = self.dequantize(quantized_sphere)
        mesh_quantized_sphere = mesh.copy()
        mesh_quantized_sphere.vertices = dequantized_sphere
        mesh_quantized_sphere_path = self.meshes_dir / f"{mesh_name}_quantized_sphere.obj"
        mesh_quantized_sphere.export(str(mesh_quantized_sphere_path))
        
        # Task 3: Dequantize, Denormalize & Error Analysis
        # Reconstruct MinMax
        dequantized_minmax = self.dequantize(quantized_minmax)
        reconstructed_minmax = self.denormalize_minmax(dequantized_minmax, params_minmax)
        errors_minmax = self.compute_errors(vertices, reconstructed_minmax)
        
        # Reconstruct Unit Sphere
        dequantized_sphere = self.dequantize(quantized_sphere)
        reconstructed_sphere = self.denormalize_unit_sphere(dequantized_sphere, params_sphere)
        errors_sphere = self.compute_errors(vertices, reconstructed_sphere)
        
        # Save reconstructed meshes
        mesh_recon_minmax = mesh.copy()
        mesh_recon_minmax.vertices = reconstructed_minmax
        mesh_recon_minmax_path = self.meshes_dir / f"{mesh_name}_reconstructed_minmax.obj"
        mesh_recon_minmax.export(str(mesh_recon_minmax_path))
        
        mesh_recon_sphere = mesh.copy()
        mesh_recon_sphere.vertices = reconstructed_sphere
        mesh_recon_sphere_path = self.meshes_dir / f"{mesh_name}_reconstructed_sphere.obj"
        mesh_recon_sphere.export(str(mesh_recon_sphere_path))
        
        # Create error plots
        self.create_error_plots(mesh_name, errors_minmax, errors_sphere)
        
        # Store metrics
        self.metrics[mesh_name] = {
            'stats': stats,
            'minmax': {
                'normalization_params': params_minmax,
                'quantization_params': qparams_minmax,
                'errors': errors_minmax
            },
            'unit_sphere': {
                'normalization_params': params_sphere,
                'quantization_params': qparams_sphere,
                'errors': errors_sphere
            }
        }
        
        print(f"\n[OK] Completed processing: {mesh_name}")
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_path = self.run_output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nSaved metrics: {metrics_path}")
    
    def generate_report(self):
        """Generate Markdown report."""
        report_path = self.run_output_dir / "REPORT.md"
        
        report_lines = [
            "# Mesh Normalization, Quantization, and Error Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Number of bins:** {self.bins}",
            "",
            "---",
            ""
        ]
        
        for mesh_name, data in self.metrics.items():
            report_lines.extend([
                f"## Mesh: {mesh_name}",
                "",
                "### Statistics",
                "",
                f"- **Number of vertices:** {data['stats']['num_vertices']}",
                "",
                "**Per-axis statistics:**",
                "",
                "| Axis | Min | Max | Mean | Std |",
                "|------|-----|-----|------|-----|"
            ])
            
            axes = ['X', 'Y', 'Z']
            for i, axis in enumerate(axes):
                report_lines.append(
                    f"| {axis} | {data['stats']['min'][i]:.6f} | "
                    f"{data['stats']['max'][i]:.6f} | "
                    f"{data['stats']['mean'][i]:.6f} | "
                    f"{data['stats']['std'][i]:.6f} |"
                )
            
            # MinMax results
            report_lines.extend([
                "",
                "### MinMax Normalization Results",
                "",
                "**Normalization Parameters:**",
                f"- Min: {data['minmax']['normalization_params']['min']}",
                f"- Max: {data['minmax']['normalization_params']['max']}",
                "",
                "**Error Metrics:**",
                "",
                "| Metric | Global | X-axis | Y-axis | Z-axis |",
                "|--------|--------|--------|--------|--------|"
            ])
            
            mse_global = data['minmax']['errors']['global']['MSE']
            mae_global = data['minmax']['errors']['global']['MAE']
            mse_axis = data['minmax']['errors']['per_axis']['MSE']
            mae_axis = data['minmax']['errors']['per_axis']['MAE']
            
            report_lines.append(
                f"| MSE | {mse_global:.6e} | {mse_axis[0]:.6e} | "
                f"{mse_axis[1]:.6e} | {mse_axis[2]:.6e} |"
            )
            report_lines.append(
                f"| MAE | {mae_global:.6e} | {mae_axis[0]:.6e} | "
                f"{mae_axis[1]:.6e} | {mae_axis[2]:.6e} |"
            )
            
            # Unit Sphere results
            report_lines.extend([
                "",
                "### Unit Sphere Normalization Results",
                "",
                "**Normalization Parameters:**",
                f"- Centroid: {data['unit_sphere']['normalization_params']['centroid']}",
                f"- Scale: {data['unit_sphere']['normalization_params']['scale']:.6f}",
                f"- Max Distance: {data['unit_sphere']['normalization_params']['max_distance']:.6f}",
                "",
                "**Error Metrics:**",
                "",
                "| Metric | Global | X-axis | Y-axis | Z-axis |",
                "|--------|--------|--------|--------|--------|"
            ])
            
            mse_global = data['unit_sphere']['errors']['global']['MSE']
            mae_global = data['unit_sphere']['errors']['global']['MAE']
            mse_axis = data['unit_sphere']['errors']['per_axis']['MSE']
            mae_axis = data['unit_sphere']['errors']['per_axis']['MAE']
            
            report_lines.append(
                f"| MSE | {mse_global:.6e} | {mse_axis[0]:.6e} | "
                f"{mse_axis[1]:.6e} | {mse_axis[2]:.6e} |"
            )
            report_lines.append(
                f"| MAE | {mae_global:.6e} | {mae_axis[0]:.6e} | "
                f"{mae_axis[1]:.6e} | {mae_axis[2]:.6e} |"
            )
            
            # Comparison
            report_lines.extend([
                "",
                "### Comparison",
                "",
                f"**Global MSE:**",
                f"- MinMax: {data['minmax']['errors']['global']['MSE']:.6e}",
                f"- Unit Sphere: {data['unit_sphere']['errors']['global']['MSE']:.6e}",
                "",
                f"**Global MAE:**",
                f"- MinMax: {data['minmax']['errors']['global']['MAE']:.6e}",
                f"- Unit Sphere: {data['unit_sphere']['errors']['global']['MAE']:.6e}",
                "",
                "### Visualizations",
                "",
                f"- Scatter plots: `{mesh_name}_scatter_plots.png`",
                f"- Error analysis: `{mesh_name}_error_analysis.png`",
                f"- MSE comparison: `{mesh_name}_mse_comparison.png`",
                "",
                "---",
                ""
            ])
        
        # Summary
        report_lines.extend([
            "## Summary",
            "",
            "This report summarizes the mesh normalization, quantization, and error analysis results.",
            "Two normalization methods were compared:",
            "",
            "1. **MinMax Normalization:** Maps coordinates directly to [0, 1] based on min/max values.",
            "2. **Unit Sphere Normalization:** Centers the mesh, scales to fit a unit sphere, then maps to [0, 1].",
            "",
            "Both methods were quantized using 1024 bins, then reconstructed to compute error metrics.",
            "",
            "### Observations",
            "",
            "- The error metrics show the reconstruction quality after quantization.",
            "- Lower MSE/MAE values indicate better preservation of the original mesh geometry.",
            "- The choice of normalization method may affect error distribution across axes.",
            ""
        ])
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Generated report: {report_path}")
    
    def run(self):
        """Run the complete pipeline."""
        # Find all OBJ files
        obj_files = list(self.input_dir.glob("*.obj"))
        
        if not obj_files:
            print(f"Warning: No .obj files found in {self.input_dir}")
            return
        
        print(f"Found {len(obj_files)} mesh file(s)")
        
        # Process each mesh
        for mesh_path in obj_files:
            try:
                self.process_mesh(mesh_path)
            except Exception as e:
                print(f"Error processing {mesh_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save metrics and generate report
        self.save_metrics()
        self.generate_report()
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed!")
        print(f"Output directory: {self.run_output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Mesh Normalization, Quantization, and Error Analysis Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="meshes",
        help="Input directory containing .obj files (default: meshes)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=1024,
        help="Number of quantization bins (default: 1024)"
    )
    
    args = parser.parse_args()
    
    processor = MeshProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bins=args.bins
    )
    
    processor.run()


if __name__ == "__main__":
    main()


