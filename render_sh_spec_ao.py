"""
Matte-clay mesh renderer: SH / SH+Spec / SH+AO / SH+Spec+AO, 4-panel, no
color/texture -- matching the reference figure Nisarg shared (2026-09-18).

No GPU or display required: shading is computed in numpy, ambient occlusion
via Open3D's CPU raycasting scene (no OpenGL/EGL/OSMesa needed), and the
image is rasterized with matplotlib's Agg backend. Runs the same way on the
Jarvis login node or a local Windows machine.

We do NOT have Nisarg's actual shading script (only reference images), so
the SH coefficients / specular / AO parameters here are a first-pass
approximation -- render one real asset, hold it next to his reference image,
and tune SH_COEFFS / SPEC_* / AO_* below to match.

Usage:
    python render_sh_spec_ao.py --mesh path/to/mesh.glb --title "prompt text" --out fig.png
    python render_sh_spec_ao.py --demo --out demo_fig.png     # synthetic test mesh, no real asset needed
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

# ---- tunable shading parameters (first pass -- tune against Nisarg's reference) ----
# Order-2 SH coefficients (9 floats, L0..L2), hand-picked for a soft overhead
# key + ambient fill, achromatic (same value used for R=G=B). L0 term kept
# high relative to the directional terms so nothing goes fully black --
# clay renders always keep some ambient floor on the unlit side.
SH_COEFFS = np.array([0.85, 0.45, 0.60, 0.15, 0.04, 0.04, 0.08, 0.04, 0.02])
AMBIENT_FLOOR = 0.22  # irradiance never drops below this fraction of peak
KEY_LIGHT_DIR = np.array([0.35, 0.55, 0.75])
KEY_LIGHT_DIR = KEY_LIGHT_DIR / np.linalg.norm(KEY_LIGHT_DIR)
SPEC_SHININESS = 24.0
SPEC_INTENSITY = 0.6
AO_SAMPLES = 64
AO_MAX_DIST = None  # None = mesh diagonal * 0.5, set below per-mesh
AO_POWER = 1.6  # >1 = more contrast/darker in occluded crevices
AO_STRENGTH = 0.85  # 0 = AO has no effect, 1 = full AO darkening
CLAY_ALBEDO = 0.85  # base matte grey (0..1), off-white like the reference
EDGE_TINT = (0.6, 0.6, 0.6, 0.12)  # very faint edge lines -- keep the surface reading as smooth clay, not wireframe


def sh9_basis(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3 * z * z - 1),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ], axis=-1)


def sh_irradiance(normals, coeffs):
    basis = sh9_basis(normals)
    raw = basis @ coeffs
    peak = coeffs[0] / 0.282095 + np.sum(np.abs(coeffs[1:]))  # rough max
    floor = AMBIENT_FLOOR * peak
    return np.clip(raw, floor, None)


def compute_ao(mesh: trimesh.Trimesh, n_samples=AO_SAMPLES, max_dist=None):
    import open3d as o3d

    if max_dist is None:
        max_dist = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 0.5

    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
    )
    scene.add_triangles(t_mesh)

    verts = mesh.vertices
    normals = mesh.vertex_normals
    eps = max_dist * 1e-3

    rng = np.random.default_rng(0)
    ao = np.zeros(len(verts))
    for i in range(len(verts)):
        n = normals[i]
        tangent = np.cross(n, [0, 1, 0] if abs(n[1]) < 0.9 else [1, 0, 0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(n, tangent)

        u1 = rng.random(n_samples)
        u2 = rng.random(n_samples)
        r = np.sqrt(u1)
        theta = 2 * np.pi * u2
        local = np.stack([r * np.cos(theta), r * np.sin(theta), np.sqrt(np.clip(1 - u1, 0, None))], axis=-1)
        dirs = local[:, 0:1] * tangent + local[:, 1:2] * bitangent + local[:, 2:3] * n

        origins = np.tile(verts[i] + n * eps, (n_samples, 1)).astype(np.float32)
        rays = np.concatenate([origins, dirs.astype(np.float32)], axis=1)
        result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        hit_dist = result["t_hit"].numpy()
        occluded = np.sum(hit_dist < max_dist)
        ao[i] = 1.0 - occluded / n_samples
    return ao


def shade_mesh(mesh: trimesh.Trimesh):
    normals = mesh.vertex_normals
    irr = sh_irradiance(normals, SH_COEFFS)
    ndoth = np.clip(normals @ KEY_LIGHT_DIR, 0, None)
    spec = SPEC_INTENSITY * (ndoth ** SPEC_SHININESS)
    ao = compute_ao(mesh)
    return irr, spec, ao


def variant_face_colors(mesh, irr, spec, ao, use_spec, use_ao):
    v = CLAY_ALBEDO * irr
    if use_ao:
        ao_display = 1.0 - AO_STRENGTH * (1.0 - ao ** AO_POWER)
        v = v * ao_display
    if use_spec:
        v = v + spec
    v = np.clip(v, 0.0, 1.0)
    face_v = v[mesh.faces].mean(axis=1)
    return np.stack([face_v, face_v, face_v, np.ones_like(face_v)], axis=-1)


def render_panel(ax, mesh, colors, elev=25, azim=35):
    poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolor=colors,
                             edgecolor=EDGE_TINT, linewidths=0.08)
    ax.add_collection3d(poly)
    b = mesh.bounds
    center = b.mean(axis=0)
    r = np.max(b[1] - b[0]) / 2 * 1.1
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def make_figure(mesh: trimesh.Trimesh, title: str, out_path: str):
    irr, spec, ao = shade_mesh(mesh)
    variants = [
        ("SH", False, False),
        ("SH+Spec", True, False),
        ("SH+AO", False, True),
        ("SH+Spec+AO", True, True),
    ]
    fig = plt.figure(figsize=(9, 9), facecolor="white")
    fig.suptitle(title, fontsize=11, y=0.98)
    # explicit grid rects -- matplotlib's tight_layout/auto-margins are
    # unreliable for 3D axes and can shrink/misplace the plotted mesh
    rects = [
        (0.02, 0.50, 0.46, 0.42), (0.52, 0.50, 0.46, 0.42),
        (0.02, 0.04, 0.46, 0.42), (0.52, 0.04, 0.46, 0.42),
    ]
    for i, (name, use_spec, use_ao) in enumerate(variants):
        ax = fig.add_axes(rects[i], projection="3d")
        colors = variant_face_colors(mesh, irr, spec, ao, use_spec, use_ao)
        render_panel(ax, mesh, colors)
        ax.set_title(name, fontsize=10, y=0.95)
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


def demo_mesh():
    # two overlapping spheres give a visible occluded crevice, unlike a
    # coarse flat-faced box, so AO differences are actually visible
    body = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    limb = trimesh.creation.icosphere(subdivisions=4, radius=0.45)
    limb.apply_translation([0.95, 0.15, 0.1])
    return trimesh.util.concatenate([body, limb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--out", type=str, default="fig.png")
    args = ap.parse_args()

    if args.demo:
        mesh = demo_mesh()
        title = args.title or "demo synthetic mesh (no real asset needed)"
    else:
        if not args.mesh:
            raise SystemExit("pass --mesh path/to/file.glb or --demo")
        mesh = trimesh.load(args.mesh, force="mesh")
        title = args.title

    make_figure(mesh, title, args.out)


if __name__ == "__main__":
    main()
