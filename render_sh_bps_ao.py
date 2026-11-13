"""
SH_BPS_AO renders, 18 views per mesh, script version of sh_bps_ao_portable.ipynb.
Run through render_sh_bps_ao.slurm.
Differences from the notebook: no install cells, EXR or plotly previews, rows flipped
instead of the upright rotation, ONLY_PREFIX env var.
"""
import os
import sys

import torch

print(f"Python: {sys.version}")
print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no CUDA GPU visible. AO (CPU) will work but the render")
    print("stage requires nvdiffrast RasterizeCudaContext and will fail.")

import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()
print("nvdiffrast CUDA rasterization context created successfully.")

MESH_DIR = os.environ.get(
    "MESH_DIR", os.path.join(os.getcwd(), "meshes")
)
OUT_DIR = os.environ.get(
    "SH_RENDER_OUT", os.path.join(os.getcwd(), "sh_render_out")
)
ENV_DIR = os.path.join(OUT_DIR, "env")

SH_MATRIX_PATH = os.path.join(ENV_DIR, "sh_irradiance_matrix_synthetic.npz")

MESH_EXTS = (".glb", ".gltf", ".ply", ".obj", ".stl", ".off", ".dae")

N_VIEWS = 18
IMAGE_SIZE = 1600
CAMERA_FOV_DEG = 40.0
CAMERA_DISTANCE = 2.2
NEAR, FAR = 0.1, 10.0

BASE_COLOR = 0.7          # achromatic diffuse albedo rho
SPEC_KS = 0.2             # Blinn-Phong specular weight
SPEC_P = 32.0             # Blinn-Phong shininess
KNEE_T = 0.9              # soft-knee transfer knee point
AO_STRENGTH = 1.0         # A_eff = 1 - AO_STRENGTH * (1 - A)

APPLY_UPRIGHT_ROT = False  # Cube3D/CubePart meshes are already +y up

N_AO_SAMPLES = 64
AO_RAY_EPS = 1e-4
AO_CHUNK_VERTS = 2048
AO_SEED = 13              # identical to ao_precompute.ipynb

for _d in (ENV_DIR,):
    os.makedirs(_d, exist_ok=True)

print(f"MESH_DIR = {MESH_DIR}")
print(f"OUT_DIR  = {OUT_DIR}")

import math
import time
import traceback

import numpy as np
import trimesh
from PIL import Image
from trimesh.ray.ray_pyembree import RayMeshIntersector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SH_CACHE = None

# Upright-orientation correction (dataset convention): 180-degree rotation
# about the x-axis, det=+1, preserves triangle winding. Applied to vertices
# at load time; vertex normals are recomputed from rotated vertices.
UPRIGHT_ROT = None


def _upright_rot():
    global UPRIGHT_ROT
    if UPRIGHT_ROT is None:
        UPRIGHT_ROT = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )
    return UPRIGHT_ROT


def load_sh():
    """Returns (M [3,4,4] tensor on DEVICE, L_lm [9,3] float32 numpy)."""
    global _SH_CACHE
    if _SH_CACHE is None:
        sh_data = np.load(SH_MATRIX_PATH)
        M = torch.from_numpy(sh_data["M"].astype(np.float32)).to(DEVICE)
        L_lm = sh_data["L_lm"].astype(np.float32)
        assert M.shape == (3, 4, 4) and L_lm.shape == (9, 3), "unexpected SH npz shapes"
        assert np.isfinite(L_lm).all(), "non-finite SH coefficients"
        _SH_CACHE = (M, L_lm)
    return _SH_CACHE

GRID_H = 512
GRID_W = 1024
L_PEAK_SHAPE = 3.0
L_FLOOR_SHAPE = 0.15
BAND_SIGMA_DEG = 28.0
COLOR_RGB = np.array([1.0, 0.98, 0.95], dtype=np.float64)
RHO = 0.7  # must match BASE_COLOR
TARGET_PEAK_LDR = 0.9  # exposed so brightest highlight sits just under saturation

C1 = 0.429043
C2 = 0.511664
C3 = 0.743125
C4 = 0.886227
C5 = 0.247708


def build_axisymmetric_radiance(h, w, l_peak, l_floor, band_sigma_deg, color_rgb):
    theta = np.pi * (np.arange(h, dtype=np.float64) + 0.5) / h
    band_sigma = np.deg2rad(band_sigma_deg)
    elevation_profile = l_floor + (l_peak - l_floor) * np.exp(
        -0.5 * ((theta - np.pi / 2.0) / band_sigma) ** 2
    )
    radiance_row = elevation_profile[:, None] * color_rgb[None, :]
    radiance = np.repeat(radiance_row[:, None, :], w, axis=1)
    return radiance.astype(np.float64)


def calibrate_exposure_scale(l_peak_shape, l_floor_shape, band_sigma_deg, color_rgb, rho, target_peak_ldr):
    """Determine the absolute radiance scale so that the brightest horizon normal,
    after the physically-correct Lambertian BRDF (rho/pi), lands at target_peak_ldr
    instead of clipping. Preserves the L_peak:L_floor shape ratio (the artistic
    choice of contrast); only the absolute exposure is calibrated.
    """
    probe_h, probe_w = 128, 256
    radiance_shape = build_axisymmetric_radiance(
        probe_h, probe_w, l_peak_shape, l_floor_shape, band_sigma_deg, color_rgb
    )
    coeffs_shape = project_environment_to_sh(radiance_shape)
    matrix_shape = sh9_to_irradiance_matrix(coeffs_shape)

    horizon_normal = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    e_horizon = irradiance_from_matrix(matrix_shape, horizon_normal)[0]
    e_max_shape = float(np.max(e_horizon))

    scale = target_peak_ldr / (e_max_shape * (rho / math.pi))
    print(
        f"[calibrate_exposure_scale] unscaled peak irradiance E_max={e_max_shape:.4f}, "
        f"unscaled LDR peak={e_max_shape * rho / math.pi:.4f}, "
        f"required exposure scale={scale:.4f}"
    )
    return scale


def equirect_directions_and_solid_angles(h, w):
    i = np.arange(h, dtype=np.float64)
    j = np.arange(w, dtype=np.float64)
    theta = np.pi * (i + 0.5) / h
    phi = 2.0 * np.pi * (j + 0.5) / w

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    sin_theta = np.sin(theta_grid)

    x = sin_theta * np.cos(phi_grid)
    y = np.cos(theta_grid)
    z = sin_theta * np.sin(phi_grid)

    d_theta = np.pi / h
    d_phi = 2.0 * np.pi / w
    solid_angle = sin_theta * d_theta * d_phi

    return x, y, z, solid_angle


def real_sh_basis(x, y, z):
    return [
        0.282095 * np.ones_like(x),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]


def project_environment_to_sh(radiance):
    h, w, num_channels = radiance.shape
    x, y, z, solid_angle = equirect_directions_and_solid_angles(h, w)
    basis = real_sh_basis(x, y, z)

    coeffs = np.zeros((9, num_channels), dtype=np.float64)
    for lm_index, y_lm in enumerate(basis):
        weight = y_lm * solid_angle
        for c in range(num_channels):
            coeffs[lm_index, c] = np.sum(radiance[:, :, c] * weight)
    return coeffs


def sh9_to_irradiance_matrix(coeffs):
    num_channels = coeffs.shape[1]
    matrix = np.zeros((num_channels, 4, 4), dtype=np.float32)
    for c in range(num_channels):
        l00 = coeffs[0, c]
        l1m1 = coeffs[1, c]
        l10 = coeffs[2, c]
        l11 = coeffs[3, c]
        l2m2 = coeffs[4, c]
        l2m1 = coeffs[5, c]
        l20 = coeffs[6, c]
        l21 = coeffs[7, c]
        l22 = coeffs[8, c]

        matrix[c] = np.array(
            [
                [C1 * l22, C1 * l2m2, C1 * l21, C2 * l11],
                [C1 * l2m2, -C1 * l22, C1 * l2m1, C2 * l1m1],
                [C1 * l21, C1 * l2m1, C3 * l20, C2 * l10],
                [C2 * l11, C2 * l1m1, C2 * l10, C4 * l00 - C5 * l20],
            ],
            dtype=np.float32,
        )
    return matrix


def irradiance_from_matrix(matrix, normals):
    n_tilde = np.concatenate([normals, np.ones((normals.shape[0], 1), dtype=np.float32)], axis=1)
    return np.einsum("vi,cij,vj->vc", n_tilde, matrix, n_tilde)


def test_azimuthal_symmetry(matrix):
    directions = {
        "+x": [1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "+z": [0.0, 0.0, 1.0],
        "-z": [0.0, 0.0, -1.0],
        "diag_xz": [0.70710678, 0.0, 0.70710678],
    }
    print("[test_azimuthal_symmetry] all vertical normals (n_y=0) should give equal irradiance:")
    values = {}
    for name, n in directions.items():
        n_arr = np.array([n], dtype=np.float32)
        e = irradiance_from_matrix(matrix, n_arr)[0]
        values[name] = e
        print(f"  n={name} {np.round(n, 3)}  E(n)={np.round(e, 4)}")

    ref = values["+x"]
    max_spread = 0.0
    for name, e in values.items():
        spread = np.max(np.abs(e - ref)) / np.maximum(np.max(np.abs(ref)), 1e-6)
        max_spread = max(max_spread, spread)
    print(f"[test_azimuthal_symmetry] max relative spread across azimuths: {max_spread:.6f}")
    return max_spread


def brute_force_irradiance(radiance, x, y, z, solid_angle, normal):
    cos_term = np.maximum(0.0, x * normal[0] + y * normal[1] + z * normal[2])
    weight = cos_term * solid_angle
    num_channels = radiance.shape[2]
    return np.array([np.sum(radiance[:, :, c] * weight) for c in range(num_channels)])


def test_bruteforce_cross_check(radiance, matrix):
    h, w, _ = radiance.shape
    x, y, z, solid_angle = equirect_directions_and_solid_angles(h, w)

    rng = np.random.default_rng(42)
    test_normals = [
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    for _ in range(4):
        v = rng.normal(size=3)
        test_normals.append(v / np.linalg.norm(v))

    print("[test_bruteforce_cross_check] normal | SH matrix E(n) | brute-force E(n) | relative error")
    max_rel_error = 0.0
    for normal in test_normals:
        e_sh = irradiance_from_matrix(matrix, normal.reshape(1, 3).astype(np.float32))[0]
        e_bf = brute_force_irradiance(radiance, x, y, z, solid_angle, normal)
        rel_error = np.abs(e_sh - e_bf) / np.maximum(np.abs(e_bf), 1e-6)
        max_rel_error = max(max_rel_error, np.max(rel_error))
        print(
            f"  n={np.round(normal, 3)}  SH={np.round(e_sh, 4)}  "
            f"BF={np.round(e_bf, 4)}  rel_err={np.round(rel_error, 4)}"
        )
    print(f"[test_bruteforce_cross_check] max relative error across all normals/channels: {max_rel_error:.4f}")
    return max_rel_error


scale = calibrate_exposure_scale(
    L_PEAK_SHAPE, L_FLOOR_SHAPE, BAND_SIGMA_DEG, COLOR_RGB, RHO, TARGET_PEAK_LDR
)
l_peak = L_PEAK_SHAPE * scale
l_floor = L_FLOOR_SHAPE * scale

print(
    f"Building synthetic axisymmetric HDRI: {GRID_H}x{GRID_W}, "
    f"L_peak={l_peak:.4f} at horizon, L_floor={l_floor:.4f} at poles, band_sigma={BAND_SIGMA_DEG} deg "
    f"(exposure-calibrated for rho={RHO}, target peak LDR={TARGET_PEAK_LDR})"
)
radiance = build_axisymmetric_radiance(GRID_H, GRID_W, l_peak, l_floor, BAND_SIGMA_DEG, COLOR_RGB)

coeffs = project_environment_to_sh(radiance)
print("SH coefficients L_lm (rows=00,1-1,10,11,2-2,2-1,20,21,22; cols=R,G,B):")
print(coeffs)

non_zonal_rows = [1, 3, 4, 5, 7, 8]
non_zonal_magnitude = np.max(np.abs(coeffs[non_zonal_rows]))
zonal_magnitude = np.max(np.abs(coeffs[[0, 2, 6]]))
print(
    f"Non-zonal (m != 0) coefficient max magnitude: {non_zonal_magnitude:.6e} "
    f"vs zonal (m=0) max magnitude: {zonal_magnitude:.6e} "
    f"(non-zonal should be ~0 for exact azimuthal symmetry, up to quadrature roundoff)"
)

matrix = sh9_to_irradiance_matrix(coeffs)
np.savez(SH_MATRIX_PATH, M=matrix, L_lm=coeffs.astype(np.float32))
print(f"Saved: {SH_MATRIX_PATH}")

test_azimuthal_symmetry(matrix)
test_bruteforce_cross_check(radiance, matrix)

def _fib_dirs(n):
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)],
        axis=1,
    )  # (x, y, z), y-up, matching equirect d_ij


def _sh_basis_9(dirs):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    return np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ], axis=1)


def dominant_light_direction():
    _, L_lm = load_sh()
    l1 = np.array([L_lm[3].mean(), L_lm[1].mean(), L_lm[2].mean()])
    l1_norm = np.linalg.norm(l1)
    dirs = _fib_dirs(2048)
    radiance_sh = _sh_basis_9(dirs) @ L_lm                     # [N,3]
    w = np.clip(radiance_sh, 0.0, None).mean(axis=1)           # [N]

    if l1_norm > 0.05 * abs(L_lm[0].mean()):
        l_analytic = l1 / l1_norm
        l_numeric = (dirs * w[:, None]).sum(axis=0)
        l_numeric = l_numeric / np.linalg.norm(l_numeric)
        cos_agree = float(l_analytic @ l_numeric)
        print(f"[spec] L1 dir = {np.round(l_analytic, 4)}, "
              f"centroid = {np.round(l_numeric, 4)}, cos = {cos_agree:.4f}")
        assert cos_agree > 0.95, "L1-analytic and numeric centroid disagree"
        return l_numeric.astype(np.float32)

    # Degenerate L1: axisymmetric/origin-centered environment. Place the
    # specular key at the brightest reconstructed direction, i.e. on the
    # horizon band where the environment's energy actually is.
    l = dirs[int(np.argmax(radiance_sh.mean(axis=1)))]
    print(f"[spec] degenerate L1 (|l1|={l1_norm:.2e}) -> argmax-radiance dir "
          f"= {np.round(l, 4)}, L(dir) = {radiance_sh.mean(axis=1).max():.4f}")
    return l.astype(np.float32)


LIGHT_DIR = dominant_light_direction()

def fibonacci_sphere_cameras(n, distance):
    indices = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    golden_angle = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden_angle * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return (np.stack([x, y, z], axis=1) * distance).astype(np.float32)


def look_at_matrix(eye, target, up_hint):
    f = target - eye
    f = f / np.linalg.norm(f)
    if abs(np.dot(f, up_hint)) > 0.999:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    s = np.cross(f, up_hint)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, eye)
    view[1, 3] = -np.dot(u, eye)
    view[2, 3] = np.dot(f, eye)
    return view


def perspective_matrix(fov_deg, aspect, near, far):
    fov_rad = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)


def build_camera_batch(n_views, distance, fov_deg):
    eyes = fibonacci_sphere_cameras(n_views, distance)
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view_matrices = np.stack(
        [look_at_matrix(eye, np.zeros(3, dtype=np.float32), up_hint) for eye in eyes], axis=0
    )
    proj = perspective_matrix(fov_deg, 1.0, NEAR, FAR)
    mvp = np.einsum("ij,bjk->bik", proj, view_matrices)
    c2w = np.zeros_like(view_matrices)
    c2w[:, :3, :3] = np.transpose(view_matrices[:, :3, :3], (0, 2, 1))
    c2w[:, :3, 3] = eyes
    c2w[:, 3, 3] = 1.0
    return view_matrices, mvp, c2w, eyes


def compute_vertex_normals(vertices, faces):
    """Face-area-weighted vertex normals. np.bincount replaces np.add.at
    (~20x faster on 300K-face meshes); accumulation runs in float64."""
    tri_v = vertices[faces]
    face_normals = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    idx = faces.reshape(-1)
    normals = np.stack([
        np.bincount(idx, weights=np.repeat(face_normals[:, c], 3),
                    minlength=vertices.shape[0])
        for c in range(3)
    ], axis=1).astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths

def cosine_hemisphere_dirs(n_samples, seed):
    """Stratified cosine-weighted directions in the local +z hemisphere."""
    s = int(math.ceil(math.sqrt(n_samples)))
    rng = np.random.default_rng(seed)
    u1 = (np.repeat(np.arange(s), s) + rng.random(s * s)) / s
    u2 = (np.tile(np.arange(s), s) + rng.random(s * s)) / s
    u1, u2 = u1[:n_samples], u2[:n_samples]
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    z = np.sqrt(np.clip(1.0 - u1, 0.0, 1.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float64)


def tangent_frames(normals):
    """Per-normal orthonormal basis (tangent, bitangent, normal)."""
    n = normals.astype(np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ref = np.where(
        np.abs(n[:, 2:3]) < 0.999,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    t = np.cross(n, ref)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(n, t)
    return t, b, n


def compute_ao(points, normals, intersector, dirs_local, eps=AO_RAY_EPS,
             chunk=AO_CHUNK_VERTS):
    """Cosine-weighted hemispherical AO at arbitrary points.

    Returns ao [N] float64 and bent_normals [N,3] float64.
    """
    t, b, n = tangent_frames(normals)
    n_pts = points.shape[0]
    ao = np.empty(n_pts, dtype=np.float64)
    bent = np.empty((n_pts, 3), dtype=np.float64)
    n_dirs = dirs_local.shape[0]

    for lo in range(0, n_pts, chunk):
        hi = min(lo + chunk, n_pts)
        k = hi - lo
        world = (
            t[lo:hi, :, None] * dirs_local[None, :, 0]
            + b[lo:hi, :, None] * dirs_local[None, :, 1]
            + n[lo:hi, :, None] * dirs_local[None, :, 2]
        )  # [k, 3, S]
        world = np.transpose(world, (0, 2, 1)).reshape(-1, 3)  # [k*S, 3]
        origins = np.repeat(points[lo:hi] + eps * n[lo:hi], n_dirs, axis=0)
        occluded = intersector.intersects_any(origins, world)
        vis = (~occluded).reshape(k, n_dirs).astype(np.float64)
        ao[lo:hi] = vis.mean(axis=1)
        bent_sum = (vis[..., None] * world.reshape(k, n_dirs, 3)).sum(axis=1)
        norms = np.linalg.norm(bent_sum, axis=1, keepdims=True)
        fallback = norms[:, 0] < 1e-9
        norms[fallback] = 1.0
        bent[lo:hi] = bent_sum / norms
        bent[lo:hi][fallback] = n[lo:hi][fallback]
    return ao, bent


def compute_mesh_ao(vertices, faces, dirs_local):
    """AO for one (already normalized/rotated) vertex-face array pair."""
    normals = compute_vertex_normals(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float64),
                           faces=faces.astype(np.int64), process=False)
    intersector = RayMeshIntersector(mesh)
    ao, bent = compute_ao(vertices.astype(np.float64), normals, intersector, dirs_local)
    return ao.astype(np.float32), bent.astype(np.float32)

_dirs_selftest = cosine_hemisphere_dirs(N_AO_SAMPLES, AO_SEED)

_sphere = trimesh.creation.icosphere(subdivisions=3)
_sphere_ao, _ = compute_ao(
    _sphere.vertices.copy(),
    compute_vertex_normals(np.asarray(_sphere.vertices, dtype=np.float64),
                           np.asarray(_sphere.faces, dtype=np.int64)),
    RayMeshIntersector(_sphere),
    _dirs_selftest,
)
_dev = float(np.max(np.abs(_sphere_ao - 1.0)))
print(f"[selftest] icosphere: min A={_sphere_ao.min():.4f}, max dev from 1 = {_dev:.4f}")
assert _dev < 0.02, "sphere AO deviates from 1 by more than 2%"

_center_ao, _ = compute_ao(
    np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]),
    RayMeshIntersector(_sphere), _dirs_selftest,
)
print(f"[selftest] sphere-center probe: A={_center_ao[0]:.4f}")
assert _center_ao[0] < 0.03, "fully occluded probe should have A ~= 0"

_quad = trimesh.Trimesh(
    vertices=np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]]),
    faces=np.array([[0, 1, 2], [0, 2, 3]]),
    process=False,
)
_quad_isect = RayMeshIntersector(_quad)
_up_ao, _ = compute_ao(np.array([[0, 0, 0.01]]), np.array([[0.0, 0.0, 1.0]]),
                       _quad_isect, _dirs_selftest)
_dn_ao, _ = compute_ao(np.array([[0, 0, 0.01]]), np.array([[0.0, 0.0, -1.0]]),
                       _quad_isect, _dirs_selftest)
print(f"[selftest] probe 0.01 above 20x20 quad: A(up)={_up_ao[0]:.4f}, A(down)={_dn_ao[0]:.4f}")
assert _up_ao[0] > 0.95 and _dn_ao[0] < 0.10, "quad self-occlusion check failed"

print("[selftest] ALL PASSED")

def soft_knee(x, t=KNEE_T):
    over = torch.clamp(x - t, min=0.0)
    return torch.where(x <= t, x, 1.0 - (1.0 - t) * torch.exp(-over / (1.0 - t)))


def render_views_ablated(vertices, faces, ao_v, n_views, image_size):
    """Render one mesh once; return shared geometry + 4 shaded rgb stacks.

    Returns:
        geom: dict of numpy arrays (normal, depth, alpha, c2w)
        rgb:  dict variant -> [n_views, H, W, 3] uint8
    """
    v = torch.from_numpy(vertices.astype(np.float32)).to(DEVICE)
    tri = torch.from_numpy(faces.astype(np.int32)).to(DEVICE)
    ones = torch.ones((v.shape[0], 1), dtype=torch.float32, device=DEVICE)
    v_h = torch.cat([v, ones], dim=1)

    view_matrices, mvp, c2w, eyes = build_camera_batch(n_views, CAMERA_DISTANCE, CAMERA_FOV_DEG)
    mvp_t = torch.from_numpy(mvp).to(DEVICE)
    view_t = torch.from_numpy(view_matrices).to(DEVICE)
    eyes_t = torch.from_numpy(eyes).to(DEVICE)                     # [V_view,3]
    light_t = torch.from_numpy(LIGHT_DIR).to(DEVICE)               # [3]

    pos_clip = torch.einsum("bij,vj->bvi", mvp_t, v_h).contiguous()
    v_view = torch.einsum("bij,vj->bvi", view_t, v_h)
    view_depth = v_view[:, :, 2:3].contiguous()

    mesh_normals = compute_vertex_normals(vertices, faces)
    normals_t = torch.from_numpy(mesh_normals.astype(np.float32)).to(DEVICE)
    ao_t = torch.from_numpy(ao_v.astype(np.float32)).to(DEVICE)

    attrs = torch.cat([
        normals_t,                       # [V,3] world normals
        v,                               # [V,3] world position
        ao_t[:, None],                   # [V,1] AO
    ], dim=1)                          # [V,7]
    attrs_batch = attrs.unsqueeze(0).expand(n_views, -1, -1).contiguous()

    rast_out, _ = dr.rasterize(glctx, pos_clip, tri, resolution=[image_size, image_size])
    mask = (rast_out[..., 3:4] > 0).float()

    depth_interp, _ = dr.interpolate(view_depth, rast_out, tri)
    depth_out = (-depth_interp) * mask

    attr_interp, _ = dr.interpolate(attrs_batch, rast_out, tri)     # [B,H,W,7]
    n_hat = torch.nn.functional.normalize(attr_interp[..., 0:3], dim=-1)
    world_pos = attr_interp[..., 3:6]
    ao_pix = attr_interp[..., 6:7].clamp(0.0, 1.0)

    normal_rgb = (n_hat * 0.5 + 0.5)
    normal_rgb = dr.antialias(normal_rgb * mask, rast_out, pos_clip, tri)

    sh_m, _ = load_sh()
    n_tilde = torch.cat([n_hat, torch.ones_like(n_hat[..., :1])], dim=-1)
    irradiance = torch.einsum("vhwi,cij,vhwj->vhwc", n_tilde, sh_m, n_tilde)
    irradiance = torch.clamp(irradiance, min=0.0)

    diffuse = irradiance * (BASE_COLOR / math.pi)                   # [B,H,W,3]
    ao_eff = 1.0 - AO_STRENGTH * (1.0 - ao_pix)                     # [B,H,W,1]

    view_dir = torch.nn.functional.normalize(
        eyes_t[:, None, None, :] - world_pos, dim=-1
    )                                                              # [B,H,W,3]
    half_v = torch.nn.functional.normalize(light_t + view_dir, dim=-1)
    spec = SPEC_KS * torch.clamp((n_hat * half_v).sum(-1, keepdim=True), 0.0, 1.0) ** SPEC_P

    rgb = {}
    rgb["SH"] = soft_knee(diffuse)
    rgb["SH_BPS"] = soft_knee(diffuse + spec)
    rgb["SH_AO"] = soft_knee(diffuse * ao_eff)
    rgb["SH_BPS_AO"] = soft_knee(diffuse * ao_eff + spec)

    geom = {
        "normal": (normal_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
        "depth": depth_out.squeeze(-1).cpu().numpy().astype(np.float32),
        "alpha": (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8),
        "camera_to_world": c2w.astype(np.float32),
        "camera_angle_x": np.float32(np.deg2rad(CAMERA_FOV_DEG)),
    }
    out = {}
    for name, img in rgb.items():
        img = dr.antialias(img * mask, rast_out, pos_clip, tri)
        out[name] = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return geom, out

ONLY_PREFIX = os.environ.get("ONLY_PREFIX", "")  # e.g. 020 for one prompt


def list_mesh_files(mesh_dir, exts=MESH_EXTS):
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(
            f"MESH_DIR does not exist: {mesh_dir}. Set the MESH_DIR env var "
            "or edit the configuration cell."
        )
    names = [
        fn for fn in sorted(os.listdir(mesh_dir))
        if fn.lower().endswith(tuple(e.lower() for e in exts)) and fn.startswith(ONLY_PREFIX)
    ]
    return [os.path.join(mesh_dir, fn) for fn in names]


def load_mesh_arrays(path):
    """Return (vertices [V,3] float32, faces [F,3] int64) for any trimesh
    file. Scenes are concatenated; non-triangle data returns None."""
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh):
        return None
    if obj.faces is None or obj.faces.shape[0] == 0:
        return None
    vertices = np.asarray(obj.vertices, dtype=np.float32)
    faces = np.asarray(obj.faces, dtype=np.int64)
    return vertices, faces


def normalize_vertices(vertices):
    """Dataset convention: bbox center to origin, max extent -> 0.98."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = float(np.max(bbox_max - bbox_min))
    scale = (1.0 / extent) * 0.98 if extent > 0 else 1.0
    return (vertices - center) * scale, center, scale


def _save_npz_atomic(path, **arrays):
    """npz via tmp file + os.replace so readers never see partial files."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def save_views(mesh_dir_out, mesh_name, rgba):
    """Write all 18 views: PNGs + one {name}_views.npz rgba stack."""
    os.makedirs(mesh_dir_out, exist_ok=True)
    for vi in range(rgba.shape[0]):
        Image.fromarray(rgba[vi, :, :, :3]).save(
            os.path.join(mesh_dir_out, f"view_{vi:02d}.png")
        )
    npz_path = os.path.join(mesh_dir_out, f"{mesh_name}_views.npz")
    _save_npz_atomic(npz_path, rgba=rgba)
    return npz_path


def mesh_outputs_complete(mesh_dir_out, mesh_name, n_views=N_VIEWS):
    npz_path = os.path.join(mesh_dir_out, f"{mesh_name}_views.npz")
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path) as d:
            if "rgba" not in d.files:
                return False
    except Exception:
        return False
    for vi in range(n_views):
        if not os.path.exists(os.path.join(mesh_dir_out, f"view_{vi:02d}.png")):
            return False
    return True

mesh_paths = list_mesh_files(MESH_DIR)
print(f"{len(mesh_paths)} mesh files found in {MESH_DIR}")
assert mesh_paths, f"no meshes in {MESH_DIR}"

_dirs_local = cosine_hemisphere_dirs(N_AO_SAMPLES, AO_SEED)
_results = []

for i, mpath in enumerate(mesh_paths):
    mesh_name = os.path.splitext(os.path.basename(mpath))[0]
    mesh_out = os.path.join(OUT_DIR, mesh_name)
    t0 = time.time()
    rec = {"mesh": mesh_name, "status": "ok", "error_msg": ""}
    try:
        if mesh_outputs_complete(mesh_out, mesh_name):
            rec["status"] = "skipped (complete)"
            _results.append(rec)
            continue
        arrays = load_mesh_arrays(mpath)
        if arrays is None:
            rec["status"] = "skipped (not a triangle mesh)"
            _results.append(rec)
            print(f"  [{i + 1}/{len(mesh_paths)}] {mesh_name}: not a triangle mesh, skipped")
            continue
        vertices, faces = arrays
        vertices, center, scale = normalize_vertices(vertices)
        if APPLY_UPRIGHT_ROT:
            vertices = vertices @ _upright_rot()
        ao_v, _bent = compute_mesh_ao(vertices, faces, _dirs_local)
        geom, rgb = render_views_ablated(vertices, faces, ao_v, N_VIEWS, IMAGE_SIZE)

        rgba = np.concatenate(
            [rgb["SH_BPS_AO"], geom["alpha"][..., None]], axis=-1
        )
        rgba = np.ascontiguousarray(rgba[:, ::-1])  # nvdiffrast row 0 is the image bottom
        npz_path = save_views(mesh_out, mesh_name, rgba)
        rec["n_verts"] = int(vertices.shape[0])
        rec["n_faces"] = int(faces.shape[0])
        rec["ao_mean"] = round(float(ao_v.mean()), 4)
        rec["processing_time_s"] = round(time.time() - t0, 3)
        print(f"  [{i + 1}/{len(mesh_paths)}] {mesh_name}: "
              f"V={rec['n_verts']} F={rec['n_faces']} "
              f"mean_A={rec['ao_mean']} t={rec['processing_time_s']}s")
    except Exception as exc:
        rec["status"] = "error"
        rec["error_msg"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    _results.append(rec)
    if (i + 1) % 64 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

_n_err = sum(1 for r in _results if r["status"] == "error")
_n_ok = sum(1 for r in _results if r["status"] == "ok")
print(f"\nDone: {_n_ok} rendered, {_n_err} errors, "
      f"{len(_results) - _n_ok - _n_err} skipped")

for r in _results:
    if r["status"] != "ok":
        continue
    mesh_out = os.path.join(OUT_DIR, r["mesh"])
    assert mesh_outputs_complete(mesh_out, r["mesh"]), (
        f"{r['mesh']}: incomplete outputs"
    )
    d = np.load(os.path.join(mesh_out, f"{r['mesh']}_views.npz"))
    assert d["rgba"].shape == (N_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 4), (
        f"{r['mesh']}: unexpected rgba shape {d['rgba'].shape}"
    )
print("All rendered artifacts verified.")
