#!/usr/bin/env python
"""
Interactive 3D visualization of Step 1 dense output using plotly.

Generates an HTML file with:
  - 3D point cloud colored by visibility count
  - Camera positions with orientation axes
  - Toggle per-camera visibility overlays
  - Hover info with coordinates and per-camera visibility details

Usage:
  cd case/CylinderDIC
  python visualize_step1_interactive.py

Opens in browser automatically.
"""

import os
import sys
import json
import webbrowser
import numpy as np
from scipy.io import loadmat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DENSE_DIR = os.path.join(SCRIPT_DIR, "calibration", "dense")
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(42)

# =========================================================================
# Data loading (same as static script)
# =========================================================================

def load_dense_data(dense_dir: str):
    ply_path = os.path.join(dense_dir, "dense_points.ply")
    with open(ply_path, "r") as f:
        lines = f.readlines()
    n_verts = 0; header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"): n_verts = int(line.split()[-1])
        if line.startswith("end_header"): header_end = i + 1; break
    pts = np.zeros((n_verts, 3), dtype=np.float64)
    for i, line in enumerate(lines[header_end:header_end + n_verts]):
        pts[i] = [float(x) for x in line.strip().split()[:3]]

    nrm_path = os.path.join(dense_dir, "dense_normals.npy")
    nrm = np.load(nrm_path) if os.path.exists(nrm_path) else None

    vis_path = os.path.join(dense_dir, "vis_mask.npy")
    vis = np.load(vis_path) if os.path.exists(vis_path) else None

    return pts, nrm, vis


def load_cameras(calib_dir: str):
    cpath = os.path.join(calib_dir, "cameras.mat")
    if not os.path.exists(cpath):
        return None
    data = loadmat(cpath)
    n_cam = int(data["num_cameras"][0, 0])

    def _extract(arr, idx, shape):
        item = arr[idx]
        if arr.dtype == object:
            flat = [float(item.flat[k].item()) for k in range(item.size)]
            return np.array(flat).reshape(shape)
        return np.array(item).reshape(shape)

    K_list = [_extract(data["K_list"], i, (3, 3)) for i in range(n_cam)]
    R_list = [_extract(data["cam_from_world_R"], i, (3, 3)) for i in range(n_cam)]
    t_raw = data["cam_from_world_t"]

    t_list = []
    for i in range(n_cam):
        if t_raw.dtype == object:
            item = t_raw[i]
            if hasattr(item, 'flat'):
                flat = [float(item.flat[k].item()) for k in range(item.size)]
            else:
                flat = np.array(item).ravel().tolist()
            t_list.append(np.array(flat).reshape(3,))
        else:
            ti = np.array(t_raw[i]).ravel()
            t_list.append(ti[:3])

    # World-space camera centers
    centers = np.array([(-R.T @ t) for R, t in zip(R_list, t_list)])

    # Camera orientation axes in world frame
    axes = np.zeros((n_cam, 3, 3))  # (cam, axis=xyz, world_xyz)
    for i in range(n_cam):
        R_c2w = R_list[i].T
        axes[i, 0] = R_c2w[:, 0]  # x-axis (right)
        axes[i, 1] = R_c2w[:, 1]  # y-axis (down)
        axes[i, 2] = R_c2w[:, 2]  # z-axis (forward)

    return {"K_list": K_list, "R_list": R_list, "t_list": t_list,
            "centers": centers, "axes": axes, "num_cameras": n_cam}


# =========================================================================
# Interactive HTML with plotly
# =========================================================================

def build_interactive_html(pts, vis, nrm, calib, output_path):
    """Build an interactive plotly figure and save to HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("[BUILD] Building interactive plotly visualization...")

    n_vis = vis.sum(axis=1)
    n_cam = vis.shape[1]

    # Subsample for rendering performance
    target = min(30000, len(pts))
    idx = np.random.choice(len(pts), target, replace=False) if len(pts) > target else np.arange(len(pts))
    pts_s = pts[idx]
    n_vis_s = n_vis[idx]
    vis_s = vis[idx]

    # ---- Trace 1: Point cloud colored by visibility count ----
    hover_text = [
        f"Pt {j}<br>X={pts_s[j,0]:.1f} Y={pts_s[j,1]:.1f} Z={pts_s[j,2]:.1f}<br>"
        f"Vis cams: {n_vis_s[j]}/{n_cam}"
        for j in range(len(pts_s))
    ]

    point_trace = go.Scatter3d(
        x=pts_s[:, 0], y=pts_s[:, 1], z=pts_s[:, 2],
        mode='markers',
        marker=dict(
            size=2.5,
            color=n_vis_s,
            colorscale='Plasma',
            cmin=0, cmax=n_cam,
            colorbar=dict(title="Visible Cameras", x=1.02),
            opacity=0.85,
        ),
        text=hover_text,
        hoverinfo='text',
        name=f'Point Cloud ({len(pts_s):,} pts)',
    )

    # ---- Trace 2: Per-camera surfaces (colored by that camera's visibility) ----
    per_cam_traces = []
    visible_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                      '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb']

    for cam_id in range(n_cam):
        mask = vis_s[:, cam_id]
        if mask.sum() == 0:
            continue
        cam_pts = pts_s[mask]
        # Downsample per-camera for performance
        if len(cam_pts) > 3000:
            sub = np.random.choice(len(cam_pts), 3000, replace=False)
            cam_pts = cam_pts[sub]

        per_cam_traces.append(go.Scatter3d(
            x=cam_pts[:, 0], y=cam_pts[:, 1], z=cam_pts[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=visible_colors[cam_id % len(visible_colors)], opacity=0.4),
            text=[f"Cam {cam_id} visible"] * len(cam_pts),
            hoverinfo='text',
            name=f'Cam {cam_id} visible',
            visible='legendonly',  # hidden by default
        ))

    # ---- Trace 3: Camera centers + orientation axes ----
    camera_center_trace = go.Scatter3d(
        x=calib["centers"][:, 0],
        y=calib["centers"][:, 1],
        z=calib["centers"][:, 2],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond',
                    line=dict(color='black', width=1)),
        text=[f'Cam {i}' for i in range(n_cam)],
        textposition='top center',
        textfont=dict(size=10, color='darkred'),
        hovertext=[f'Camera {i}<br>Center: ({calib["centers"][i,0]:.1f}, '
                    f'{calib["centers"][i,1]:.1f}, {calib["centers"][i,2]:.1f})'
                   for i in range(n_cam)],
        hoverinfo='text',
        name='Cameras',
    )

    # ---- Trace 4: Camera axis cones ----
    axis_traces = []
    arrow_scale = np.linalg.norm(pts_s.std(axis=0)) * 0.12
    axis_colors = {'X': 'red', 'Y': 'green', 'Z': 'blue'}

    for i in range(n_cam):
        center = calib["centers"][i]
        for a_idx, (ax_name, color) in enumerate(axis_colors.items()):
            direction = calib["axes"][i, a_idx]
            end = center + direction * arrow_scale
            axis_traces.append(go.Scatter3d(
                x=[center[0], end[0]],
                y=[center[1], end[1]],
                z=[center[2], end[2]],
                mode='lines',
                line=dict(color=color, width=3),
                hoverinfo='text',
                hovertext=f'Cam {i} {ax_name}-axis',
                name=f'Cam {i} {ax_name}',
                showlegend=(i == 0),  # legend once per color
                legendgroup=ax_name,
            ))

    # ---- Figure assembly ----
    fig = go.Figure(data=[point_trace, camera_center_trace] + axis_traces + per_cam_traces)

    # ---- Layout ----
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Step 1 Dense Output — Interactive Viewer</b><br>"
                f"<sub>{len(pts_s):,} points sampled from {len(pts):,} · "
                f"Mean visible: {n_vis_s.mean():.1f}/{n_cam} · "
                f"Hover for details · Click legend to toggle</sub>"
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

    # ---- Axis equal-ish aspect ratio ----
    ranges = np.ptp(pts_s, axis=0)
    max_range = ranges.max()
    center = pts_s.mean(axis=0)
    for axis_name in ['xaxis', 'yaxis', 'zaxis']:
        fig.update_scenes({f'{axis_name}_range': [
            center[{'xaxis': 0, 'yaxis': 1, 'zaxis': 2}[axis_name]] - max_range / 2,
            center[{'xaxis': 0, 'yaxis': 1, 'zaxis': 2}[axis_name]] + max_range / 2,
        ]})

    # ---- Save ----
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    print(f"  HTML size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print(f"  → {output_path}")


# =========================================================================
# Also save a lightweight CSV for external tools
# =========================================================================

def export_for_paraview(pts, vis, nrm, report_dir):
    """Export point cloud as CSV that ParaView/CloudCompare can read directly."""
    print("[EXPORT] ParaView-compatible CSV...")

    # Subsample to 50K for manageable file size
    target = min(50000, len(pts))
    idx = np.random.choice(len(pts), target, replace=False)
    pts_s = pts[idx]
    n_vis = vis.sum(axis=1)[idx]

    header = "x,y,z,visible_cameras"
    if nrm is not None:
        header += ",nx,ny,nz"
        nrm_s = nrm[idx]
        data = np.column_stack([pts_s, n_vis.reshape(-1, 1), nrm_s])
    else:
        data = np.column_stack([pts_s, n_vis.reshape(-1, 1)])

    csv_path = os.path.join(report_dir, "dense_points_vis.csv")
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.4f')
    print(f"  → {csv_path} ({os.path.getsize(csv_path)/1024:.0f} KB, {len(pts_s)} pts)")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  Interactive Step 1 Visualization (plotly)")
    print("=" * 60)

    pts, nrm, vis = load_dense_data(DENSE_DIR)
    calib = load_cameras(CALIB_DIR)

    html_path = os.path.join(REPORT_DIR, "step1_viewer.html")
    build_interactive_html(pts, vis, nrm, calib, html_path)

    export_for_paraview(pts, vis, nrm, REPORT_DIR)

    # Auto-open in browser
    print(f"\n[OPEN] Opening in browser...")
    webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")

    print(f"\n[DONE]")
    print(f"  Interactive viewer: report/step1_viewer.html")
    print(f"  ParaView CSV:       report/dense_points_vis.csv")
    print(f"\n  Controls:")
    print(f"    Left-drag:  Rotate")
    print(f"    Right-drag: Pan")
    print(f"    Scroll:     Zoom")
    print(f"    Click legend: Toggle cameras/layers")
    print(f"    Hover:      Point details")


if __name__ == "__main__":
    main()
