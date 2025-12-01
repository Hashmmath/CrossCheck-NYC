"""
Stage B Visualization
=====================

Creates visualizations for Stage B network-level analysis results.

Usage:
    python visualize_stage_b.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_stage_b_results():
    """Load Stage B results from JSON."""
    results_path = Path("outputs/network/stage_b_results.json")
    
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        print("Run 'python run_complete_pipeline.py' first")
        return None
    
    with open(results_path) as f:
        return json.load(f)

def create_stage_b_dashboard(results: dict, save_path: Path):
    """Create a comprehensive Stage B dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Stage B: Network-Level Analysis Dashboard", fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Panel 1: Detection Coverage (Top Left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    detected = results.get('total_detected_pixels', 0)
    gt = results.get('total_gt_pixels', 1)
    coverage = results.get('detection_coverage', 0)
    
    # Pie chart showing coverage
    if detected > 0 and gt > 0:
        overlap = min(detected, gt)
        detected_only = max(0, detected - gt)
        missed = max(0, gt - detected)
        
        sizes = [overlap, detected_only, missed]
        labels = ['Matched', 'Over-detection', 'Missed']
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        explode = (0.05, 0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    
    ax1.set_title('Detection Coverage Analysis', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 2: Pixel Statistics (Top Middle)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    categories = ['Detected\nPixels', 'Ground Truth\nPixels']
    values = [detected, gt]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Pixel Count', fontsize=11)
    ax2.set_title('Detection vs Ground Truth', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel 3: A→B Linkage Metrics (Top Right)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    metrics = results.get('aggregate_metrics', {})
    
    metric_names = ['IoU', 'F1', 'Precision', 'Recall']
    metric_values = [
        metrics.get('mean_iou', 0),
        metrics.get('mean_f1', 0),
        metrics.get('mean_precision', 0),
        metrics.get('mean_recall', 0)
    ]
    
    colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax3.barh(metric_names, metric_values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, metric_values):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlim(0, 1.15)
    ax3.set_xlabel('Score', fontsize=11)
    ax3.set_title('Stage A Metrics (A→B Linkage)', fontsize=12, fontweight='bold')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # =========================================================================
    # Panel 4: Network Quality Gauge (Bottom Left)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate overall quality score
    quality_score = np.mean([
        metrics.get('mean_iou', 0),
        metrics.get('mean_f1', 0),
        min(coverage, 1.0) if coverage else 0
    ])
    
    # Create gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arc
    ax4.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=20, solid_capstyle='round')
    
    # Quality arc
    quality_theta = np.linspace(0, np.pi * quality_score, 100)
    color = '#2ecc71' if quality_score > 0.7 else '#f39c12' if quality_score > 0.4 else '#e74c3c'
    ax4.plot(r * np.cos(quality_theta), r * np.sin(quality_theta), color, linewidth=20, solid_capstyle='round')
    
    # Add text
    ax4.text(0, 0.3, f'{quality_score:.1%}', ha='center', va='center', fontsize=24, fontweight='bold')
    ax4.text(0, -0.1, 'Network Quality', ha='center', va='center', fontsize=12)
    
    quality_label = 'Good' if quality_score > 0.7 else 'Moderate' if quality_score > 0.4 else 'Needs Improvement'
    ax4.text(0, -0.3, quality_label, ha='center', va='center', fontsize=14, 
             color=color, fontweight='bold')
    
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-0.5, 1.2)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Overall Network Quality Score', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 5: A→B Linkage Explanation (Bottom Middle)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    linkage_text = """
    A → B Linkage Analysis
    ━━━━━━━━━━━━━━━━━━━━━━━━
    
    Stage A (Pixel-Level):
    • IoU: {iou:.1%} pixel overlap
    • F1: {f1:.1%} detection accuracy
    
    Stage B (Network-Level):
    • Coverage: {cov:.1%}
    • Quality Score: {qual:.1%}
    
    Interpretation:
    {interp}
    """.format(
        iou=metrics.get('mean_iou', 0),
        f1=metrics.get('mean_f1', 0),
        cov=min(coverage, 1.0) if coverage else 0,
        qual=quality_score,
        interp=get_interpretation(quality_score, metrics)
    )
    
    ax5.text(0.1, 0.9, linkage_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # Panel 6: Baseline Comparison (Bottom Right)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    baseline_features = results.get('baseline_features', 0)
    
    comparison_data = {
        'Our Detection': detected,
        'LION Baseline': baseline_features * 10  # Approximate pixel equivalent
    }
    
    bars = ax6.bar(comparison_data.keys(), comparison_data.values(), 
                   color=['#3498db', '#95a5a6'], edgecolor='black', alpha=0.8)
    
    ax6.set_ylabel('Scale (relative)', fontsize=11)
    ax6.set_title('Detection vs Official Baseline', fontsize=12, fontweight='bold')
    ax6.grid(True, axis='y', alpha=0.3)
    
    # Add note
    ax6.text(0.5, -0.15, f'LION contains {baseline_features:,} street features',
             transform=ax6.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved Stage B dashboard to: {save_path}")


def get_interpretation(quality_score, metrics):
    """Get interpretation text based on scores."""
    if quality_score > 0.7:
        return "Strong pixel-to-network quality.\nDetections are reliable for routing."
    elif quality_score > 0.5:
        return "Moderate quality. Some gaps\nmay affect network connectivity."
    else:
        return "Quality needs improvement.\nSignificant gaps in detection."


def create_ab_linkage_diagram(results: dict, save_path: Path):
    """Create A→B linkage concept diagram."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    fig.suptitle("A → B Linkage: From Pixels to Networks", fontsize=16, fontweight='bold')
    
    # Stage A Box
    stage_a_box = mpatches.FancyBboxPatch((0.5, 3.5), 3.5, 2, 
                                           boxstyle="round,pad=0.1",
                                           facecolor='#3498db', alpha=0.3,
                                           edgecolor='#2980b9', linewidth=2)
    ax.add_patch(stage_a_box)
    ax.text(2.25, 5.2, "STAGE A", ha='center', fontsize=14, fontweight='bold', color='#2980b9')
    ax.text(2.25, 4.8, "Pixel-Level Evaluation", ha='center', fontsize=11)
    
    metrics = results.get('aggregate_metrics', {})
    stage_a_text = f"""
    IoU: {metrics.get('mean_iou', 0):.3f}
    F1: {metrics.get('mean_f1', 0):.3f}
    Precision: {metrics.get('mean_precision', 0):.3f}
    Recall: {metrics.get('mean_recall', 0):.3f}
    """
    ax.text(2.25, 4.2, stage_a_text, ha='center', fontsize=10, fontfamily='monospace')
    
    # Arrow
    ax.annotate('', xy=(5.5, 4.5), xytext=(4.2, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.text(4.85, 4.8, "informs", ha='center', fontsize=10, style='italic')
    
    # Stage B Box
    stage_b_box = mpatches.FancyBboxPatch((6, 3.5), 3.5, 2,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#2ecc71', alpha=0.3,
                                           edgecolor='#27ae60', linewidth=2)
    ax.add_patch(stage_b_box)
    ax.text(7.75, 5.2, "STAGE B", ha='center', fontsize=14, fontweight='bold', color='#27ae60')
    ax.text(7.75, 4.8, "Network-Level Quality", ha='center', fontsize=11)
    
    coverage = results.get('detection_coverage', 0)
    stage_b_text = f"""
    Coverage: {min(coverage, 1.0):.1%}
    Detected: {results.get('total_detected_pixels', 0):,} px
    Baseline: {results.get('baseline_features', 0):,} features
    """
    ax.text(7.75, 4.2, stage_b_text, ha='center', fontsize=10, fontfamily='monospace')
    
    # Bottom: Impact boxes
    # Good pixels → Good network
    ax.annotate('', xy=(2.25, 3.2), xytext=(2.25, 3.5),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))
    
    impact_a = mpatches.FancyBboxPatch((0.5, 1), 3.5, 2,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#ecf0f1', alpha=0.8,
                                        edgecolor='gray', linewidth=1)
    ax.add_patch(impact_a)
    ax.text(2.25, 2.7, "Pixel Quality Impact", ha='center', fontsize=11, fontweight='bold')
    ax.text(2.25, 2.2, "• High IoU → Accurate boundaries", ha='center', fontsize=9)
    ax.text(2.25, 1.8, "• Good calibration → Reliable confidence", ha='center', fontsize=9)
    ax.text(2.25, 1.4, "• High recall → Few missed crosswalks", ha='center', fontsize=9)
    
    # Network impact
    ax.annotate('', xy=(7.75, 3.2), xytext=(7.75, 3.5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    impact_b = mpatches.FancyBboxPatch((6, 1), 3.5, 2,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#ecf0f1', alpha=0.8,
                                        edgecolor='gray', linewidth=1)
    ax.add_patch(impact_b)
    ax.text(7.75, 2.7, "Network Quality Impact", ha='center', fontsize=11, fontweight='bold')
    ax.text(7.75, 2.2, "• Connected pedestrian paths", ha='center', fontsize=9)
    ax.text(7.75, 1.8, "• Usable for routing applications", ha='center', fontsize=9)
    ax.text(7.75, 1.4, "• Matches official city data", ha='center', fontsize=9)
    
    # Connection arrow at bottom
    ax.annotate('', xy=(5.8, 2), xytext=(4.2, 2),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2))
    ax.text(5, 2.3, "enables", ha='center', fontsize=10, style='italic', color='#9b59b6')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved A→B linkage diagram to: {save_path}")


def main():
    """Create all Stage B visualizations."""
    
    print("=" * 60)
    print("STAGE B VISUALIZATION")
    print("=" * 60)
    
    # Load results
    results = load_stage_b_results()
    
    if results is None:
        return
    
    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("\nCreating Stage B dashboard...")
    create_stage_b_dashboard(results, output_dir / "stage_b_dashboard.png")
    
    print("Creating A→B linkage diagram...")
    create_ab_linkage_diagram(results, output_dir / "ab_linkage_diagram.png")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  • {output_dir / 'stage_b_dashboard.png'}")
    print(f"  • {output_dir / 'ab_linkage_diagram.png'}")


if __name__ == "__main__":
    main()