#!/usr/bin/env python
"""
Visualize Step 3 displacement field output.

Generates:
  1. 3D displacement field colored by magnitude (per step)
  2. Displacement components (u, v, w) side-by-side
  3. Deformed vs undeformed overlay comparison
  4. Displacement magnitude histogram + statistics
  5. Multi-step evolution summary (if multiple load steps)
  6. Interactive HTML viewer (plotly)

Usage:
  cd case/CylinderDIC
  python visualize_displacement.py

  # Or from project root:
  python case/CylinderDIC/visualize_displacement.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

# --- Resolve paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

DPI = 150
np.random.seed(42)


# =========================================================================
# Data loading
# =========================================================================

def load_displacement_data(results_dir: str):
    """Load all Step 3 displacement outputs."""
    print(f"[LOAD] results_dir = {results_dir}")

    # Metadata
    meta_path = os.path.join(results_dir, "results_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    # Reference points
    ref_path = os.path.join(results_dir, "ref_points.npy")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Not found: {ref_path}")
    ref_pts = np.load(ref_path)
    print(f"  Ref points: {ref_pts.shape}")

    # Discover displacement steps
    disp_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith("disp_step") and f.endswith(".npy")
    ])
    if not disp_files:
        raise FileNotFoundError(f"No disp_step*.npy found in {results_dir}")

    n_steps = len(disp_files)
    print(f"  Load steps: {n_steps}")

    # Load all displacement fields
    disp_fields = {}
    def_fields = {}
    for f in disp_files:
        step_num = int(f.replace("disp_step", "").replace(".npy", ""))
        disp_fields[step_num] = np.load(os.path.join(results_dir, f))

        def_f = f"def_points_step{step_num:03d}.npy"
        def_path = os.path.join(results_dir, def_f)
        if os.path.exists(def_path):
            def_fields[step_num] = np.load(def_path)

    return ref_pts, disp_fields, def_fields, meta


def subsample(pts, *arrays, target=15000):
    """Randomly subsample to target points."""
    if len(pts) <= target:
        return (pts,) + arrays
    idx = np.random.choice(len(pts), target, replace=False)
    idx.sort()
    return (pts[idx],) + tuple(a[idx] if a is not None else None for a in arrays)


# =========================================================================
# Color normalization (robust to outliers)
# =========================================================================

def robust_vlim(values, clip_percentile=98):
    """Return vmin, vmax that clips extreme outliers."""
    vmin = np.percentile(values, 2)
    vmax = np.percentile(values, clip_percentile)
    return vmin, vmax


# =========================================================================
# Figure 1: 3D displacement magnitude (per step)
# =========================================================================

def fig1_displacement_3d(ref_pts, disp_fields, report_dir):
    """3D point cloud colored by displacement magnitude."""
    print("[FIG 1] 3D displacement magnitude...")

    n_steps = len(disp_fields)
    n_cols = min(3, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5.5 * n_rows),
        subplot_kw={'projection': '3d'},
    )
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    steps_sorted = sorted(disp_fields.keys())

    for idx, step in enumerate(steps_sorted):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        disp = disp_fields[step]
        mag = np.linalg.norm(disp, axis=1)

        pts_s, mag_s = subsample(ref_pts, mag, target=12000)
        vmin, vmax = robust_vlim(mag, clip_percentile=98)

        scatter = ax.scatter(
            pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
            c=mag_s, cmap='turbo', s=1.5, alpha=0.85,
            vmin=vmin, vmax=vmax,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label('|Φ| (mm)', fontsize=9)

        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f'Step {step} — Displacement Magnitude\n'
                     f'mean={mag.mean():.3f}  max={mag.max():.3f} mm',
                     fontsize=10, fontweight='bold')
        ax.view_init(elev=25, azim=-55)

    # Hide unused axes
    for idx in range(n_steps, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Step 3 — Displacement Field (3D View)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join(report_dir, "step3_displacement_3d.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 2: Displacement components (u, v, w) per step
# =========================================================================

def fig2_displacement_components(ref_pts, disp_fields, report_dir):
    """Show u, v, w components on orthogonal projections for the first step."""
    print("[FIG 2] Displacement components...")

    step = sorted(disp_fields.keys())[0]  # first step
    disp = disp_fields[step]
    mag = np.linalg.norm(disp, axis=1)

    pts_s, disp_s, mag_s = subsample(ref_pts, disp, mag, target=15000)

    components = [
        (0, 'u (X-displacement)', 'RdBu_r'),
        (1, 'v (Y-displacement)', 'RdBu_r'),
        (2, 'w (Z-displacement)', 'RdBu_r'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for i, (comp_idx, title, cmap) in enumerate(components):
        ax = axes[i]
        values = disp_s[:, comp_idx]
        vlim = max(abs(values.min()), abs(values.max()))
        vlim = min(vlim, np.percentile(np.abs(values), 98))

        sc = ax.scatter(
            pts_s[:, 0], pts_s[:, 2],  # X-Z projection
            c=values, cmap=cmap, s=2, alpha=0.8,
            vmin=-vlim, vmax=vlim,
        )
        plt.colorbar(sc, ax=ax, label=f'{title.split()[0]} (mm)', shrink=0.8)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')

    # Magnitude histogram
    ax = axes[3]
    ax.hist(mag, bins=50, color='steelblue', edgecolor='white', alpha=0.85, density=True)
    ax.axvline(mag.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {mag.mean():.4f} mm')
    ax.axvline(np.median(mag), color='orange', linestyle='--', linewidth=1.5,
               label=f'Median = {np.median(mag):.4f} mm')
    ax.set_xlabel('Displacement Magnitude (mm)')
    ax.set_ylabel('Density')
    ax.set_title(f'|Φ| Distribution (Step {step})', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Step 3 — Displacement Components (Step {step}, X-Z view)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(report_dir, "step3_displacement_components.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 3: Deformed vs undeformed overlay
# =========================================================================

def fig3_deformed_comparison(ref_pts, disp_fields, def_fields, report_dir):
    """Overlay reference (gray) and deformed (colored by displacement) point clouds."""
    print("[FIG 3] Deformed vs undeformed comparison...")

    step = sorted(disp_fields.keys())[0]
    disp = disp_fields[step]

    pts_s, disp_s = subsample(ref_pts, disp, target=10000)
    mag_s = np.linalg.norm(disp_s, axis=1)
    vmin, vmax = robust_vlim(mag_s, clip_percentile=98)

    # Deformed points
    x_def_s = pts_s + disp_s

    fig = plt.figure(figsize=(16, 7))

    # ---- Subplot 1: Overlay (reference gray + deformed colored) ----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Reference in light gray
    ax1.scatter(
        pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
        c='lightgray', s=0.5, alpha=0.4, label='Reference',
    )
    # Deformed in color
    sc = ax1.scatter(
        x_def_s[:, 0], x_def_s[:, 1], x_def_s[:, 2],
        c=mag_s, cmap='turbo', s=2, alpha=0.85,
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(sc, ax=ax1, shrink=0.6, pad=0.08, label='|Φ| (mm)')
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Step {step}: Reference + Deformed Overlay', fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.view_init(elev=25, azim=-55)

    # ---- Subplot 2: Displacement vectors (quiver on X-Z plane) ----
    ax2 = fig.add_subplot(1, 2, 2)

    # Subsample further for quiver clarity
    n_q = min(800, len(pts_s))
    idx_q = np.random.choice(len(pts_s), n_q, replace=False)
    pts_q = pts_s[idx_q]
    disp_q = disp_s[idx_q]
    mag_q = mag_s[idx_q]

    # Draw displacement vectors on X-Z plane
    q = ax2.quiver(
        pts_q[:, 0], pts_q[:, 2],
        disp_q[:, 0], disp_q[:, 2],  # u, w components in X-Z
        mag_q, cmap='turbo', scale=1.0, width=0.003,
        alpha=0.7, clim=(vmin, vmax),
    )
    plt.colorbar(q, ax=ax2, label='|Φ| (mm)', shrink=0.8)
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Z (mm)')
    ax2.set_title(f'Step {step}: Displacement Vectors (X-Z plane, {n_q} pts)',
                  fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Step 3 — Deformed Shape Comparison', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join(report_dir, "step3_deformed_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 4: Multi-step evolution (if n_steps > 1)
# =========================================================================

def fig4_multi_step_evolution(ref_pts, disp_fields, report_dir):
    """Show how displacement magnitude evolves across load steps."""
    print("[FIG 4] Multi-step evolution...")

    steps_sorted = sorted(disp_fields.keys())
    n_steps = len(steps_sorted)

    if n_steps < 2:
        print("  [SKIP] Only 1 load step, no evolution to show")
        return

    # Compute per-step statistics
    stats = []
    for step in steps_sorted:
        mag = np.linalg.norm(disp_fields[step], axis=1)
        stats.append({
            "step": step,
            "mean": mag.mean(),
            "median": np.median(mag),
            "max": mag.max(),
            "std": mag.std(),
        })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Subplot 1: Mean & max displacement vs step ----
    ax = axes[0, 0]
    steps_x = [s["step"] for s in stats]
    ax.plot(steps_x, [s["mean"] for s in stats], 'o-', color='steelblue',
            linewidth=2, markersize=8, label='Mean |Φ|')
    ax.fill_between(steps_x,
                    [s["mean"] - s["std"] for s in stats],
                    [s["mean"] + s["std"] for s in stats],
                    alpha=0.2, color='steelblue')
    ax2_ = ax.twinx()
    ax2_.bar(steps_x, [s["max"] for s in stats], alpha=0.35, color='orangered',
             label='Max |Φ|')
    ax.set_xlabel('Load Step'); ax.set_ylabel('Mean ± Std (mm)')
    ax2_.set_ylabel('Max (mm)')
    ax.set_title('Displacement Evolution', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps_x)

    # ---- Subplot 2: Displacement histogram per step (overlaid) ----
    ax = axes[0, 1]
    cmap = plt.cm.viridis
    for i, step in enumerate(steps_sorted):
        mag = np.linalg.norm(disp_fields[step], axis=1)
        color = cmap(i / max(n_steps - 1, 1))
        ax.hist(mag, bins=40, alpha=0.35, color=color,
                label=f'Step {step}', density=True)
    ax.set_xlabel('|Φ| (mm)'); ax.set_ylabel('Density')
    ax.set_title('Displacement Distribution per Step', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 3: Displacement magnitude on X-Z projection (last step) ----
    ax = axes[1, 0]
    last_step = steps_sorted[-1]
    disp_last = disp_fields[last_step]
    mag_last = np.linalg.norm(disp_last, axis=1)
    pts_s, mag_s = subsample(ref_pts, mag_last, target=12000)
    vmin, vmax = robust_vlim(mag_s, clip_percentile=98)
    sc = ax.scatter(pts_s[:, 0], pts_s[:, 2], c=mag_s, cmap='turbo',
                    s=2, alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='|Φ| (mm)', shrink=0.8)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
    ax.set_title(f'Final State — Step {last_step} (X-Z view)', fontweight='bold')
    ax.set_aspect('equal')

    # ---- Subplot 4: Incremental displacement between consecutive steps ----
    ax = axes[1, 1]
    for i in range(1, n_steps):
        step_prev = steps_sorted[i - 1]
        step_cur = steps_sorted[i]
        delta = np.linalg.norm(disp_fields[step_cur] - disp_fields[step_prev], axis=1)
        ax.hist(delta, bins=40, alpha=0.5, density=True,
                label=f'Δ Step {step_prev}→{step_cur}')
    ax.set_xlabel('Incremental |ΔΦ| (mm)'); ax.set_ylabel('Density')
    ax.set_title('Incremental Displacement Between Steps', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Step 3 — Multi-Step Evolution ({n_steps} load steps)',
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    path = os.path.join(report_dir, "step3_multistep_evolution.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 5: Summary statistics card
# =========================================================================

def fig5_summary_card(ref_pts, disp_fields, meta, report_dir):
    """Generate a summary statistics figure."""
    print("[FIG 5] Summary statistics...")

    steps_sorted = sorted(disp_fields.keys())
    first_disp = disp_fields[steps_sorted[0]]
    mag = np.linalg.norm(first_disp, axis=1)

    stats = {
        "Reference Points": f"{len(ref_pts):,}",
        "Load Steps": str(len(steps_sorted)),
        "Bounding Box X": f"[{ref_pts[:,0].min():.1f}, {ref_pts[:,0].max():.1f}] mm",
        "Bounding Box Y": f"[{ref_pts[:,1].min():.1f}, {ref_pts[:,1].max():.1f}] mm",
        "Bounding Box Z": f"[{ref_pts[:,2].min():.1f}, {ref_pts[:,2].max():.1f}] mm",
    }

    # Per-step stats
    for step in steps_sorted:
        d = disp_fields[step]
        m = np.linalg.norm(d, axis=1)
        stats[f"Step {step} — Mean |Φ|"] = f"{m.mean():.4f} mm"
        stats[f"Step {step} — Max  |Φ|"] = f"{m.max():.4f} mm"
        stats[f"Step {step} — Std  |Φ|"] = f"{m.std():.4f} mm"

    # Component-wise stats (first step)
    for comp, name in [(0, 'u'), (1, 'v'), (2, 'w')]:
        vals = first_disp[:, comp]
        stats[f"  {name} range"] = f"[{vals.min():.4f}, {vals.max():.4f}] mm"

    fig, ax = plt.subplots(1, 1, figsize=(10, max(10, len(stats) * 0.35)))
    ax.axis('off')

    y = 0.97
    ax.text(0.5, y, "Step 3 Displacement Field — Summary Statistics",
            transform=ax.transAxes, fontsize=15, fontweight='bold',
            ha='center', va='center', fontfamily='monospace')

    for i, (key, val) in enumerate(stats.items()):
        y = 0.93 - i * 0.025
        is_header = key.startswith("Step ")
        ax.text(0.08, y, key, transform=ax.transAxes, fontsize=10,
                fontfamily='monospace',
                fontweight='bold' if is_header else 'normal',
                va='center')
        ax.text(0.58, y, val, transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', va='center',
                color='darkblue' if is_header else 'black')

    ax.text(0.5, 0.02, f"Results directory: results/",
            transform=ax.transAxes, fontsize=9, ha='center', va='center',
            fontfamily='monospace', color='gray')

    path = os.path.join(report_dir, "step3_summary.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Interactive HTML viewer (plotly)
# =========================================================================

def build_interactive_html(ref_pts, disp_fields, def_fields, report_dir):
    """Build an interactive plotly HTML with displacement field."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[HTML] plotly not installed, skipping interactive viewer")
        return

    print("[HTML] Building interactive plotly viewer...")

    steps_sorted = sorted(disp_fields.keys())
    n_steps = len(steps_sorted)
    n_pts = len(ref_pts)

    # Subsample for rendering
    target = min(25000, n_pts)
    idx = np.random.choice(n_pts, target, replace=False) if n_pts > target else np.arange(n_pts)
    pts_s = ref_pts[idx]

    # ---- Trace 1: Reference points (faint gray) ----
    ref_trace = go.Scatter3d(
        x=pts_s[:, 0], y=pts_s[:, 1], z=pts_s[:, 2],
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.3),
        name='Reference (undeformed)',
        hoverinfo='skip',
    )

    traces = [ref_trace]

    # ---- Trace 2+: Deformed points per step, colored by displacement magnitude ----
    for step in steps_sorted:
        disp_s = disp_fields[step][idx]
        mag_s = np.linalg.norm(disp_s, axis=1)
        x_def = pts_s + disp_s

        vmin, vmax = robust_vlim(mag_s, clip_percentile=98)

        hover_text = [
            f"Step {step}<br>"
            f"X={x_def[j,0]:.2f} Y={x_def[j,1]:.2f} Z={x_def[j,2]:.2f}<br>"
            f"u={disp_s[j,0]:.4f} v={disp_s[j,1]:.4f} w={disp_s[j,2]:.4f}<br>"
            f"|Φ|={mag_s[j]:.4f} mm"
            for j in range(len(pts_s))
        ]

        # Only show first step by default
        visible = (step == steps_sorted[0])

        traces.append(go.Scatter3d(
            x=x_def[:, 0], y=x_def[:, 1], z=x_def[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,
                color=mag_s,
                colorscale='Turbo',
                cmin=vmin, cmax=vmax,
                colorbar=dict(
                    title=f"|Φ| Step {step} (mm)",
                    x=1.02,
                ) if step == steps_sorted[0] else None,
                opacity=0.85,
            ),
            text=hover_text,
            hoverinfo='text',
            name=f'Step {step} (deformed)',
            visible=(True if visible else 'legendonly'),
        ))

    # ---- Figure assembly ----
    fig = go.Figure(data=traces)

    n_visible = min(3, n_steps)

    # Create visibility dropdown for steps
    if n_steps > 1:
        buttons = []
        # "All steps" button
        buttons.append(dict(
            label='All Steps',
            method='update',
            args=[{'visible': [True] + [True] * n_steps},
                  {'title': f'Displacement Field — All {n_steps} Steps'}],
        ))
        # Individual step buttons
        for i, step in enumerate(steps_sorted):
            visibility = [True] + [False] * n_steps  # ref always on
            visibility[1 + i] = True  # only this step's deformed
            buttons.append(dict(
                label=f'Step {step}',
                method='update',
                args=[{'visible': visibility},
                      {'title': f'Displacement Field — Step {step}'}],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type='dropdown',
                buttons=buttons,
                x=1.05, y=1.0,
                xanchor='left', yanchor='top',
            )],
        )

    # ---- Layout ----
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Step 3 — Displacement Field Viewer</b><br>"
                f"<sub>{target:,} points · {n_steps} load steps · "
                f"Hover for details · Legend to toggle layers</sub>"
            ),
            x=0.5, xanchor='center',
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1400, height=900,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=1.02,
            font=dict(size=9),
            itemsizing='constant',
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # Equal-ish aspect ratio
    ranges = np.ptp(pts_s, axis=0)
    max_range = ranges.max()
    center = pts_s.mean(axis=0)
    for axis_name in ['xaxis', 'yaxis', 'zaxis']:
        dim = {'xaxis': 0, 'yaxis': 1, 'zaxis': 2}[axis_name]
        fig.update_scenes({f'{axis_name}_range': [
            center[dim] - max_range / 2,
            center[dim] + max_range / 2,
        ]})

    # Save
    html_path = os.path.join(report_dir, "step3_displacement_viewer.html")
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    print(f"  HTML: {os.path.getsize(html_path) / 1024:.0f} KB → {html_path}")


# =========================================================================
# Export CSV for ParaView / CloudCompare
# =========================================================================

def export_csv(ref_pts, disp_fields, report_dir):
    """Export displacement field as CSV for external tools."""
    print("[EXPORT] ParaView-compatible CSV...")

    steps_sorted = sorted(disp_fields.keys())
    n_pts = len(ref_pts)

    # Subsample if needed
    target = min(50000, n_pts)
    if n_pts > target:
        idx = np.random.choice(n_pts, target, replace=False)
        idx.sort()
        pts_s = ref_pts[idx]
    else:
        pts_s = ref_pts
        idx = np.arange(n_pts)

    # Build CSV with all steps
    header = "x,y,z"
    data_cols = [pts_s]

    for step in steps_sorted:
        disp_s = disp_fields[step][idx]
        mag_s = np.linalg.norm(disp_s, axis=1).reshape(-1, 1)
        header += f",u_step{step},v_step{step},w_step{step},mag_step{step}"
        data_cols.append(disp_s)
        data_cols.append(mag_s)

    data = np.column_stack(data_cols)

    csv_path = os.path.join(report_dir, "displacement_field.csv")
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f"  → {csv_path} ({os.path.getsize(csv_path)/1024:.0f} KB, {len(pts_s)} pts)")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  Step 3 Displacement Field Visualization")
    print("=" * 60)

    # Check results exist
    if not os.path.isdir(RESULTS_DIR):
        print(f"\n[ERROR] Results directory not found: {RESULTS_DIR}")
        print(f"Run 'python run.py --steps 3' first to generate displacement data.")
        sys.exit(1)

    ref_pts, disp_fields, def_fields, meta = load_displacement_data(RESULTS_DIR)

    # Static figures
    fig1_displacement_3d(ref_pts, disp_fields, REPORT_DIR)
    fig2_displacement_components(ref_pts, disp_fields, REPORT_DIR)
    fig3_deformed_comparison(ref_pts, disp_fields, def_fields, REPORT_DIR)
    fig4_multi_step_evolution(ref_pts, disp_fields, REPORT_DIR)
    fig5_summary_card(ref_pts, disp_fields, meta, REPORT_DIR)

    # Interactive HTML
    build_interactive_html(ref_pts, disp_fields, def_fields, REPORT_DIR)

    # CSV export
    export_csv(ref_pts, disp_fields, REPORT_DIR)

    print(f"\n[DONE] All figures saved to {REPORT_DIR}/")
    print(f"  → step3_displacement_3d.png        3D displacement magnitude per step")
    print(f"  → step3_displacement_components.png u, v, w components + histogram")
    print(f"  → step3_deformed_comparison.png     Deformed vs reference overlay")
    print(f"  → step3_multistep_evolution.png     Multi-step evolution (if >1 step)")
    print(f"  → step3_summary.png                 Statistics card")
    print(f"  → step3_displacement_viewer.html    Interactive 3D viewer (browser)")
    print(f"  → displacement_field.csv            ParaView/CloudCompare export")
    print(f"\n  Open interactive viewer:")
    import webbrowser
    html_path = os.path.join(REPORT_DIR, "step3_displacement_viewer.html")
    webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
