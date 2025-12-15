import os
import glob
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import math
# import subprocess
import shutil
import copy
# import xatlas
import imageio
# import lpips
import torch
import torch.nn.functional as F
import random
import numpy as np
# from tqdm import tqdm
from PIL import Image
import utils3d
import json
from v2m4_trellis.pipelines import TrellisImageTo3DPipeline
from v2m4_trellis.utils import render_utils, postprocessing_utils
from v2m4_trellis.utils.general_utils import *
from v2m4_trellis.representations.mesh import MeshExtractResult
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
# from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from natsort import ns, natsorted
import trimesh
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes
# from largesteps.parameterize import from_differential, to_differential
# from largesteps.geometry import compute_matrix
# from largesteps.optimize import AdamUniform
from v2m4_trellis.utils.render_utils import rotation_6d_to_matrix
# import nvdiffrast.torch as dr
import argparse
from natsort import ns, natsorted
import dill
import tripoSG.app as tripoSG_app
from tripoSG.triposg.scripts.briarmbg import BriaRMBG
from tripoSG.triposg.triposg.pipelines.pipeline_triposg import TripoSGPipeline
from tripoSG.mv_adapter.scripts.inference_ig2mv_sdxl import prepare_pipeline
# Try to import texture module (requires GLIBC 2.29+)
try:
    from tripoSG.texture import TexturePipeline, ModProcessConfig
    TEXTURE_AVAILABLE = True
except (ImportError, OSError) as e:
    TEXTURE_AVAILABLE = False
    print(f"Warning: TexturePipeline not available ({e}). Texture features will be disabled.")
    # Create dummy classes
    class TexturePipeline:
        def __init__(self, *args, **kwargs):
            pass
    class ModProcessConfig:
        def __init__(self, *args, **kwargs):
            pass
import craftsman.app as craftsman_app
from craftsman.pipeline import CraftsManPipeline


def face_area_consistency_loss(original_vertices, deformed_vertices, faces):
    """
    Ensures that face areas remain the same between the original and deformed meshes.
    
    Args:
        original_vertices: (V, 3) Tensor of original vertex positions.
        deformed_vertices: (V, 3) Tensor of deformed vertex positions.
        faces: (F, 3) Tensor containing face indices.
    
    Returns:
        Scalar loss value enforcing area preservation.
    """
    def compute_face_areas(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        return 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)  # (F,)

    original_areas = compute_face_areas(original_vertices, faces)
    deformed_areas = compute_face_areas(deformed_vertices, faces)

    return torch.mean((deformed_areas - original_areas) ** 2)  # MSE between areas

def edge_length_consistency_loss(original_vertices, deformed_vertices, faces):
    """
    Ensures that edge lengths remain close to their original values.
    
    Args:
        original_vertices: (V, 3) Tensor of original vertex positions.
        deformed_vertices: (V, 3) Tensor of deformed vertex positions.
        faces: (F, 3) Tensor containing face indices.
    
    Returns:
        Scalar loss value enforcing edge length preservation.
    """
    def compute_edge_lengths(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        return torch.cat([
            torch.norm(v1 - v0, dim=1, keepdim=True),  # Edge v0-v1
            torch.norm(v2 - v1, dim=1, keepdim=True),  # Edge v1-v2
            torch.norm(v0 - v2, dim=1, keepdim=True)   # Edge v2-v0
        ], dim=1).view(-1)  # Flatten

    original_lengths = compute_edge_lengths(original_vertices, faces)
    deformed_lengths = compute_edge_lengths(deformed_vertices, faces)

    return torch.mean((deformed_lengths - original_lengths) ** 2)  # MSE between edge lengths


def outputs_to_files_for_blender(input_glb_folder):
    # Input and output paths
    output_npy_path = input_glb_folder + "/output_vertex_offsets.npy"
    output_texture_path = input_glb_folder + "/output_texture.png"  # Path to save the texture

    # Get all GLB files and sort them by time order
    glb_files = sorted([f for f in os.listdir(input_glb_folder) if f.endswith("_texture_consistency_sample.glb")])
    glb_files = natsorted(glb_files, alg=ns.PATH)

    # Read the first frame as a reference
    first_mesh = trimesh.load(os.path.join(input_glb_folder, glb_files[0]), process=False)
    ref_vertices = np.array(first_mesh.geometry['geometry_0'].vertices)

    # copy and rename the first frame using shutil, so output_mesh.glb is same as the (0)_texture_consistency_sample.glb
    shutil.copy(os.path.join(input_glb_folder, glb_files[0]), os.path.join(input_glb_folder, "output_mesh.glb"))

    # Try to save the texture (only executed the first time)
    if hasattr(first_mesh.geometry['geometry_0'].visual, "material") and hasattr(first_mesh.geometry['geometry_0'].visual.material, "baseColorTexture"):
        texture = first_mesh.geometry['geometry_0'].visual.material.baseColorTexture
        texture.save(output_texture_path)

    # Store vertex offsets for all frames
    vertex_offsets = []

    for glb_file in glb_files:
        mesh = trimesh.load(os.path.join(input_glb_folder, glb_file), process=False)
        current_vertices = np.array(mesh.geometry['geometry_0'].vertices)
        offset = current_vertices - ref_vertices
        vertex_offsets.append(offset)

        ref_vertices = current_vertices

    # Convert to NumPy array (num_frames, num_vertices, 3)
    vertex_offsets = np.array(vertex_offsets)
    np.save(output_npy_path, vertex_offsets)

def compute_edges_and_weights(faces, V):
    """
    Compute edges (edge list) and edge_weights (weights based on edge lengths)
    :param faces: Triangle mesh indices (M, 3)
    :param V: Vertex coordinates (N, 3)
    :return: edges (E, 2), edge_weights (E,)
    """
    # Retrieve the edges of the triangles
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)  # (3M, 2)

    # Remove duplicate edges (undirected graph)
    edges = torch.sort(edges, dim=1)[0]  # Sort each edge's endpoints to ensure undirected consistency
    edges = torch.unique(edges, dim=0)  # Remove duplicates

    # Compute edge lengths
    edge_lengths = torch.norm(V[edges[:, 0]] - V[edges[:, 1]], dim=1)

    # Compute weights based on edge lengths (to prevent division by zero)
    edge_weights = 1.0 / (edge_lengths + 1e-8)

    return edges, edge_weights


def arap_loss(V, V_opt, faces):
    """
    Compute parallel ARAP Loss using the full neighborhood to compute R
    :param V: Original vertex positions (N, 3)
    :param V_opt: Deformed vertex positions (N, 3)
    :param faces: Triangle indices of the mesh (M, 3)
    :return: ARAP Loss
    """
    # Compute edges and edge_weights
    edges, edge_weights = compute_edges_and_weights(faces, V)

    # Compute displacement before and after deformation for each edge
    V_i = V[edges[:, 0]]  # (E, 3)
    V_j = V[edges[:, 1]]  # (E, 3)
    V_opt_i = V_opt[edges[:, 0]]  # (E, 3)
    V_opt_j = V_opt[edges[:, 1]]  # (E, 3)

    # Compute local transformation matrices S_i of the original mesh
    S_i = (V_j - V_i).unsqueeze(-1) @ (V_opt_j - V_opt_i).unsqueeze(1)  # (E, 3, 3)

    # Compute local S matrices for each vertex (N, 3, 3)
    N = V.shape[0]  # Number of vertices
    S = torch.zeros((N, 3, 3), device=V.device)
    counts = torch.zeros(N, device=V.device)

    # Accumulate neighborhood contributions
    S.index_add_(0, edges[:, 0], S_i * edge_weights.view(-1, 1, 1))
    S.index_add_(0, edges[:, 1], S_i * edge_weights.view(-1, 1, 1))  # Accumulate for the other endpoint as well
    counts.index_add_(0, edges[:, 0], edge_weights)
    counts.index_add_(0, edges[:, 1], edge_weights)

    # Normalize S (to prevent numerical instability)
    S = S / (counts.view(-1, 1, 1) + 1e-8)

    # Compute batch SVD
    U, _, Vh = torch.linalg.svd(S)  # (N, 3, 3)
    R = U @ Vh  # (N, 3, 3)

    # Compute ARAP Loss
    arap_term = (V_opt_j - V_opt_i) - torch.bmm(R[edges[:, 0]], (V_j - V_i).unsqueeze(-1)).squeeze(-1)
    loss = torch.mean(edge_weights * torch.norm(arap_term, dim=-1) ** 2)

    return loss


def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_mesh_without_texture(mesh):
    """Return a copy of the input mesh with texture and per-vertex colors removed."""
    mesh_no_tex = mesh.deepcopy() if hasattr(mesh, "deepcopy") else copy.deepcopy(mesh)

    if getattr(mesh_no_tex, "vertex_attrs", None) is not None and mesh_no_tex.vertex_attrs.shape[1] >= 3:
        mesh_no_tex.vertex_attrs = mesh_no_tex.vertex_attrs.clone()
        mesh_no_tex.vertex_attrs[:, :3] = 0.7

    if hasattr(mesh_no_tex, "texture"):
        mesh_no_tex.texture = None
    if hasattr(mesh_no_tex, "uv"):
        mesh_no_tex.uv = None

    return mesh_no_tex


def align_mesh_bbox_to_x_axis(mesh):
    """
    Align the mesh so that the longer side of its horizontal (XY plane) bounding box
    is parallel to the global X axis. The mesh is rotated in-place around the Z axis.

    Args:
        mesh: MeshExtractResult-like object with `vertices` attribute of shape (N, 3).

    Returns:
        The input mesh after alignment (rotation applied in-place).
    """
    if mesh is None or not hasattr(mesh, "vertices") or mesh.vertices is None:
        return mesh

    vertices = mesh.vertices
    if vertices.numel() == 0 or vertices.shape[1] < 2:
        return mesh

    device = vertices.device
    dtype = vertices.dtype

    xy = vertices[:, :2]
    xy_center = xy.mean(dim=0, keepdim=True)
    centered_xy = xy - xy_center

    if torch.allclose(centered_xy, torch.zeros_like(centered_xy)):
        return mesh

    cov = centered_xy.t().matmul(centered_xy)
    num_points = centered_xy.shape[0]
    if num_points > 0:
        cov = cov / num_points

    eigvals, eigvecs = torch.linalg.eigh(cov)
    longest_axis_idx = torch.argmax(eigvals)
    principal_axis = eigvecs[:, longest_axis_idx]

    if torch.allclose(principal_axis.abs().sum(), torch.tensor(0.0, device=device, dtype=dtype)):
        return mesh

    angle = torch.atan2(principal_axis[1], principal_axis[0])
    rotation_angle = -angle  # rotate to align with +X

    cos_angle = torch.cos(rotation_angle)
    sin_angle = torch.sin(rotation_angle)

    rot2d = torch.stack(
        [
            torch.stack([cos_angle, -sin_angle]),
            torch.stack([sin_angle, cos_angle]),
        ]
    )
    rot = torch.eye(3, device=device, dtype=dtype)
    rot[:2, :2] = rot2d.to(device=device, dtype=dtype)

    center3d = vertices.mean(dim=0, keepdim=True)
    centered_vertices = vertices - center3d
    rotated_vertices = centered_vertices.matmul(rot.t()) + center3d
    mesh.vertices = rotated_vertices

    if getattr(mesh, "vertex_attrs", None) is not None and mesh.vertex_attrs.shape[1] >= 6:
        normals = mesh.vertex_attrs[:, 3:6]
        rotated_normals = normals.matmul(rot.t())
        mesh.vertex_attrs[:, 3:6] = F.normalize(rotated_normals, dim=1)

    if hasattr(mesh, "face_normal"):
        mesh.face_normal = mesh.comput_face_normals(mesh.vertices, mesh.faces)

    return mesh

def apply_lighting_nvdiffrast(target_mesh, params, ambient_color=[0.2, 0.2, 0.2], diffuse_color=[1.0, 1.0, 1.0], specular_color=[0.2, 0.2, 0.2], light_type='point', light_location=None, light_direction=None):
    device = target_mesh.vertices.device

    lighted_mesh = copy.deepcopy(target_mesh)

    if lighted_mesh.vertex_attrs is not None and lighted_mesh.vertex_attrs.shape[1] >= 6:
        normals = lighted_mesh.vertex_attrs[:, 3:6]
    else:
        normals = lighted_mesh.comput_v_normals(lighted_mesh.vertices, lighted_mesh.faces)
        if lighted_mesh.vertex_attrs is None:
            lighted_mesh.vertex_attrs = torch.zeros((lighted_mesh.vertices.shape[0], normals.shape[1] + 3), device=device)
        lighted_mesh.vertex_attrs[:, 3:3 + normals.shape[1]] = normals

    normals = F.normalize(normals, dim=1)
    base_color = lighted_mesh.vertex_attrs[:, :3].clamp(0.0, 1.0).clone()

    ambient = torch.tensor(ambient_color, dtype=torch.float32, device=device)
    diffuse = torch.tensor(diffuse_color, dtype=torch.float32, device=device)
    specular = torch.tensor(specular_color, dtype=torch.float32, device=device)

    if light_type == 'directional':
        if light_direction is None or isinstance(light_direction, str):
            if light_direction == 'up' or light_direction == None:
                light_dir = [0, 0, -1.0]
            elif light_direction == 'front':
                light_dir = [0, 1.0, 0]
            elif light_direction == 'back':
                light_dir = [0, -1.0, 0]
            elif light_direction == 'left':
                light_dir = [1.0, 0, 0]
            elif light_direction == 'right':
                light_dir = [-1.0, 0, 0]
            else:
                print(f"Invalid light direction: {light_direction}, using default direction")
                light_dir = [0, 0, -1.0]
        light_dir_tensor = torch.tensor(light_dir, dtype=torch.float32, device=device)
        light_dir_tensor = F.normalize(light_dir_tensor, dim=0)
        light_vec = -light_dir_tensor.unsqueeze(0).expand_as(normals)
        attenuation = torch.ones_like(light_vec[:, :1])
    else:
        if light_location is None or isinstance(light_location, str):
            mesh_center = (lighted_mesh.vertices.min(dim=0).values + lighted_mesh.vertices.max(dim=0).values) / 2
            if light_location == 'up' or light_location == None:
                light_loc_tensor = mesh_center + torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device)
            elif light_location == 'back':
                light_loc_tensor = mesh_center + torch.tensor([0, 1.0, 0], dtype=torch.float32, device=device)
            elif light_location == 'front':
                light_loc_tensor = mesh_center - torch.tensor([0, 1.0, 0], dtype=torch.float32, device=device)
            elif light_location == 'right':
                light_loc_tensor = mesh_center + torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device)
            elif light_location == 'left':
                light_loc_tensor = mesh_center - torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device)
            else:
                print(f"Invalid light location: {light_location}, using default location")
                light_loc_tensor = mesh_center + torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device)
        to_light = light_loc_tensor.unsqueeze(0) - lighted_mesh.vertices
        distance = torch.norm(to_light, dim=1, keepdim=True).clamp_min(1e-6)
        light_vec = to_light / distance
        attenuation = 1.0 / (distance ** 2)

    diffuse_intensity = torch.clamp((normals * F.normalize(light_vec, dim=1)).sum(dim=1, keepdim=True), min=0.0) * attenuation

    yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
    yaw_tensor = torch.tensor(yaw, dtype=torch.float32, device=device)
    pitch_tensor = torch.tensor(pitch, dtype=torch.float32, device=device)
    r_tensor = torch.tensor(r, dtype=torch.float32, device=device)

    camera_origin = torch.stack([
        torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
        torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
        torch.sin(pitch_tensor)
    ], dim=0) * r_tensor

    view_vec = F.normalize(camera_origin.unsqueeze(0) - lighted_mesh.vertices, dim=1)
    reflect_dir = F.normalize(2 * (normals * light_vec).sum(dim=1, keepdim=True) * normals - light_vec, dim=1)
    specular_intensity = torch.clamp((reflect_dir * view_vec).sum(dim=1, keepdim=True), min=0.0) ** 32 * attenuation

    ambient_term = ambient.view(1, 3) * base_color
    diffuse_term = diffuse_intensity * diffuse.view(1, 3) * base_color
    specular_term = specular_intensity * specular.view(1, 3)

    shaded = torch.clamp(ambient_term + diffuse_term + specular_term, 0.0, 1.0)
    lighted_mesh.vertex_attrs[:, :3] = shaded

    return lighted_mesh

def render_rotated_model(mesh, base_name, output_path, params, rotation_angle_deg=30, suffix="", 
                         light_type='point', light_location=None, light_direction=None,
                         ambient_color=None, diffuse_color=None, specular_color=None):
    """
    Rotate the model along its own center for a specified angle in yaw and save the rendered image.
    
    Args:
        mesh: MeshExtractResult object containing the mesh to render
        base_name: Base name for the output file
        output_path: Path to save the rendered image
        params: Camera parameters (yaw, pitch, r, lookat_x, lookat_y, lookat_z)
        rotation_angle_deg: Rotation angle in degrees (default: 30)
        suffix: Suffix to add to the filename (e.g., "_textured" or "_no_texture")
        light_type: Type of light - 'point' or 'directional' (default: 'point')
        light_location: Location of point light [x, y, z] (default: [0.0, 0.0, -1.0])
        light_direction: Direction of directional light [x, y, z] (default: [0.0, 0.0, -1.0])
        ambient_color: Ambient light color [r, g, b] (default: [0.5, 0.5, 0.5])
        diffuse_color: Diffuse light color [r, g, b] (default: [1.0, 1.0, 1.0])
        specular_color: Specular light color [r, g, b] (default: [1.0, 1.0, 1.0])
    
    Returns:
        Rendered image array
    """
    # Extract camera parameters
    yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
    
    # Get the mesh center (mean of min max of vertices)
    mesh_center = (mesh.vertices.min(dim=0).values + mesh.vertices.max(dim=0).values) / 2
    # mesh_center = mesh.vertices.mean(dim=0)
    
    # Translate vertices to center around origin
    vertices_centered = mesh.vertices - mesh_center
    
    # Convert rotation angle from degrees to radians
    rotation_angle_rad = math.radians(rotation_angle_deg)
    
    # Create rotation matrix for yaw rotation around Z-axis
    # Yaw rotation: rotation around the vertical axis (Z-axis)
    cos_yaw = math.cos(rotation_angle_rad)
    sin_yaw = math.sin(rotation_angle_rad)
    rotation_matrix = torch.tensor([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ], dtype=torch.float32).cuda()
    
    # Rotate vertices
    vertices_rotated = vertices_centered @ rotation_matrix.T
    
    # Translate back to original position
    vertices_rotated = vertices_rotated + mesh_center
    
    # Create a temporary mesh with rotated vertices
    rotated_mesh = copy.deepcopy(mesh)
    rotated_mesh.vertices = vertices_rotated
    
    # Update vertex normals if they exist
    if hasattr(rotated_mesh, 'face_normal'):
        rotated_mesh.face_normal = rotated_mesh.comput_face_normals(vertices_rotated, rotated_mesh.faces)
    if hasattr(rotated_mesh, 'vertex_attrs') and rotated_mesh.vertex_attrs is not None and rotated_mesh.vertex_attrs.shape[1] > 3:
        rotated_mesh.vertex_attrs[:, 3:] = rotated_mesh.comput_v_normals(vertices_rotated, rotated_mesh.faces)
    
    # Get extrinsics and intrinsics using the existing function
    # Keep the camera position the same (use original params)
    fov = 40
    from v2m4_trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], [r], [fov])

    if extr.dim() == 2:
        extr = extr.unsqueeze(0)

    if intr.dim() == 3:
        intr_batch = intr
        intr = intr[0]
    else:
        intr_batch = intr.unsqueeze(0)
    
    # Update lookat point to account for rotation around mesh center
    # lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
    # lookat_centered = lookat - mesh_center
    # lookat_rotated = lookat_centered @ rotation_matrix.T
    # updated_lookat = lookat_rotated + mesh_center
    updated_lookat = mesh_center
    
    # Update extrinsics to use the rotated lookat point
    yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
    pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
    r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
    
    orig = torch.stack([
        torch.sin(yaw_tensor) * torch.cos(pitch_tensor),
        torch.cos(yaw_tensor) * torch.cos(pitch_tensor),
        torch.sin(pitch_tensor),
    ], dim=1).squeeze() * r_tensor
    
    extr = utils3d.torch.extrinsics_look_at(orig.unsqueeze(0), updated_lookat.unsqueeze(0), torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda())
    
    resolution = 512
    rend_img = None
    
    from v2m4_trellis.utils.render_utils import render_frames
    render_options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

    
    result = render_frames(rotated_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
  
    # Save the rendered image
    suffix_str = f"_{suffix}" if suffix else ""
    output_filename = output_path + "/" + base_name + f"_rotated_{rotation_angle_deg}deg{suffix_str}_no_light.png"
    imageio.imsave(output_filename, rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='up')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_up_point_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='front')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_front_point_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='back')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_back_point_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='left')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_left_point_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='right')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_right_point_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='up')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_up_directional_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='front')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_front_directional_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='back')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_back_directional_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='left')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_left_directional_light.png'), rend_img)

    lighted_mesh = apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='right')
    result = render_frames(lighted_mesh, extr, intr_batch, render_options)
    rend_img = result['color'][0]
    imageio.imsave(output_filename.replace('_no_light.png', '_right_directional_light.png'), rend_img)

def render_video_and_glbs(outputs, base_name, output_path, init_extrinsics=None, fix_geometry=False, inplace_mesh_change=False):
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0 if fix_geometry else 0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        init_extrinsics=init_extrinsics, # Initial extrinsics for the camera
        fill_holes=False if fix_geometry else True, # Fill holes in the mesh
        return_watertight=inplace_mesh_change, # Return a watertight mesh (no texture)
        bake_v_color=True if inplace_mesh_change else False, # Bake vertex colors into the texture
    )
    glb.export(output_path + "/" + base_name + "_sample.glb")

    if inplace_mesh_change:
        # rotate as glb has a rotation pre-processing
        outputs['mesh'][0].vertices = torch.from_numpy(glb.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])).float().cuda()
        outputs['mesh'][0].faces = torch.from_numpy(glb.faces).long().cuda()

        vertex_normals = torch.from_numpy(glb.vertex_normals)
        outputs['mesh'][0].vertex_attrs = vertex_normals.repeat(1, 2).float().cuda()
        outputs['mesh'][0].vertex_attrs[:, :3] = torch.from_numpy(glb.visual.vertex_colors)[:, :3] / 255.
        
        outputs['mesh'][0].face_normal = outputs['mesh'][0].comput_face_normals(outputs['mesh'][0].vertices, outputs['mesh'][0].faces)

    return glb.visual

def parse_args():
    parser = argparse.ArgumentParser(description='V2M4 Pipeline')
    parser.add_argument('--root', type=str, default='', help='Root directory of the dataset')
    parser.add_argument('--output', type=str, default='', help='Output directory of the results')
    parser.add_argument('--N', type=int, default=1, help='Total number of parallel processes')
    parser.add_argument('--n', type=int, default=0, help='Index of the current process')
    parser.add_argument('--model', type=str, default='Hunyuan', help='Base model, TRELLIS, Craftsman, TripoSG or Hunyuan2.0', choices=['TRELLIS', 'Hunyuan', 'TripoSG', 'Craftsman'])
    # parser.add_argument('--baseline', action='store_true', help='Run the baseline model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip', type=int, default=5, help='Skip every N frames for large movement of the object (default: 5)')
    parser.add_argument('--use_vggt', action='store_true', help='Use VGGT for camera search, otherwise use dust3R (default: True)')
    # parser.add_argument('--use_tracking', action='store_true', help='Use point tracking for mesh registration guidance (!!New Feature!!)')
    # parser.add_argument('--tracking_camera_radius', type=int, default=8, help='Adapt the camera radius to ensure the object motion is within the camera view (this argument is only used when --use_tracking is set)')
    # parser.add_argument('--blender_path', type=str, default='blender-4.2.1-linux-x64/', help='Path to the Blender executable')
    parser.add_argument('--max_faces', type=int, default=10000, help='Maximum number of faces for the generated mesh (default: 10000). Lower value can speed up the process for all the 3D generation models but not affect TRELLIS, which leverages a different processing pipeline')
    # parser.add_argument('--disable_temporal_refinement',default=True, type=bool, help='Disable temporal refinement. When set, the process will finish after generating meshes for each frame, skipping topology consistency, texture consistency, and mesh interpolation.')
    parser.add_argument('--light_type', type=str, default='point', help='Type of light for PyTorch3D rendering: point or directional (default: point)', choices=['point', 'directional'])
    parser.add_argument('--light_location', type=float, nargs=3, default=None, help='Location of point light [x, y, z] (default: [0.0, 0.0, -1.0])')
    parser.add_argument('--light_direction', type=float, nargs=3, default=None, help='Direction of directional light [x, y, z] (default: [0.0, 0.0, -1.0])')
    parser.add_argument('--ambient_color', type=float, nargs=3, default=None, help='Ambient light color [r, g, b] (default: [0.5, 0.5, 0.5])')
    parser.add_argument('--diffuse_color', type=float, nargs=3, default=None, help='Diffuse light color [r, g, b] (default: [1.0, 1.0, 1.0])')
    parser.add_argument('--specular_color', type=float, nargs=3, default=None, help='Specular light color [r, g, b] (default: [1.0, 1.0, 1.0])')
    return parser.parse_args()

def get_folder_size(folder):
    """Returns the number of image files in the given folder."""
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


if __name__ == "__main__":
    args = parse_args()
    root = "examples" if args.root == "" else args.root
    default_output_root = "results_examples"
    output_root = default_output_root if args.output == "" else args.output

    os.makedirs(output_root, exist_ok=True)

    args_json_path = os.path.join(output_root, f"args_process_{args.n}.json")
    with open(args_json_path, "w") as args_file:
        json.dump(vars(args), args_file, indent=2)
    print(f"Saved arguments to {args_json_path}")

    animations = [os.path.join(root, dir) for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]

    # filter out animations where the corresponding output folder has file "output_animation.glb"
    # if not args.baseline:
    #     animations = [anim for anim in animations if not os.path.exists(os.path.join(default_output_root if args.output == "" else args.output, anim.split("/")[-1], "output_animation.glb"))]
    # Filter out animations where the corresponding output folder has file "_baseline_sample.glb"
    # animations = [anim for anim in animations if not os.path.exists(os.path.join("results_benchmark_final_seed42" if args.output == "" else args.output, anim.split("/")[-1], "0001_baseline_sample.glb"))]

    # sort
    animations = natsorted(animations, alg=ns.PATH)

    '''Parallel Processing - Average assignment according to the image numbers within each folder'''
    # Get folder sizes
    folder_sizes = [(anim, get_folder_size(anim)) for anim in animations]

    # Sort folders by size in descending order (largest first)
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    # Assign folders to processes in a balanced way
    assignments = [[] for _ in range(args.N)]
    workload = [0] * args.N  # Track workload for each process

    for folder, size in folder_sizes:
        # Assign the folder to the process with the least workload
        min_index = workload.index(min(workload))
        assignments[min_index].append(folder)
        workload[min_index] += size

    # Get the folders assigned to the current process
    assigned_animations = assignments[args.n]

    print(f"Process {args.n} assigned {len(assigned_animations)} folders with total images: {sum(get_folder_size(f) for f in assigned_animations)}")

    # Load a pipeline from a model folder or a Hugging Face model hub.
    if args.model == "TRELLIS":
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
    elif args.model == "Hunyuan":
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
        pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    elif args.model == "TripoSG" or args.model == "Craftsman":
        checkpoints_dir = "./models/checkpoints/"

        if args.model == "TripoSG":
            RMBG_PRETRAINED_MODEL = f"{checkpoints_dir}/RMBG-1.4"

            pipeline_rmbg_net = BriaRMBG.from_pretrained(RMBG_PRETRAINED_MODEL).to("cuda")
            pipeline_rmbg_net.eval()

            TRIPOSG_PRETRAINED_MODEL = f"{checkpoints_dir}/TripoSG"
            pipeline_triposg_pipe = TripoSGPipeline.from_pretrained(TRIPOSG_PRETRAINED_MODEL).to("cuda", torch.float16)
        elif args.model == "Craftsman":
            checkpoints_dir_CraftsMan = f"/{checkpoints_dir}/craftsman-DoraVAE"
            pipeline_crafts = CraftsManPipeline.from_pretrained(checkpoints_dir_CraftsMan, device="cuda", torch_dtype=torch.bfloat16) # bf16 for fast inference

        pipeline_mv_adapter_pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=None,
            num_views=6,
            device="cuda",
            dtype=torch.float16,
        )

        pipeline_texture = TexturePipeline(
            upscaler_ckpt_path=f"{checkpoints_dir}/RealESRGAN_x2plus.pth",
            inpaint_ckpt_path=f"{checkpoints_dir}/big-lama.pt",
            device="cuda",
        )

        mod_config = ModProcessConfig(view_upscale=True, inpaint_mode="view")

    # All the models in pipeline are disabled requiring_grad.
    if args.model == "TRELLIS":
        for model in pipeline.models.values():
            for param in model.parameters():
                param.requires_grad = False
    elif args.model == "Hunyuan":
        pipeline.model.requires_grad = False
        pipeline.vae.requires_grad = False
        pipeline.conditioner.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.feature_extractor.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.text_encoder.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.unet.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.vae.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.unet.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.vae.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.text_encoder.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.feature_extractor.requires_grad = False

    for animation in assigned_animations:
        source_path = animation
        output_path = os.path.join(output_root, animation.split("/")[-1])

        print("/n/n ============= Start processing: ", animation, " =============/n")

        # New folder for the output
        os.makedirs(output_path, exist_ok=True)

        # Fix the seed for reproducibility
        seed = args.seed
        seed_torch(seed)
            
        imgs_list = os.listdir(source_path)
        # exclude folders
        imgs_list = [img for img in imgs_list if not os.path.isdir(source_path + "/" + img)]
        imgs_list = natsorted(imgs_list, alg=ns.PATH)

        existing_outputs = os.listdir(output_path)
        # exclude folders
        existing_outputs = [img for img in existing_outputs if not os.path.isdir(output_path + "/" + img)]
        existing_outputs = natsorted(existing_outputs, alg=ns.PATH)

        # outputs_list = []
        # base_name_list = []
        # extrinsics_list = []
        # visual_list = []
        params = None
        for ind, img in enumerate(imgs_list):            
            # Skip every N frames for large movement of the object
            # if not args.baseline and ind % args.skip != 0 and ind != len(imgs_list) - 1:
            if ind % args.skip != 0 and ind != len(imgs_list) - 1: # processing every N frames and the last frame
                continue

            # Load an image
            image = Image.open(source_path + "/" + img)

            # Get base name of the image
            base_name = image.filename.split("/")[-1].split(".")[0]

            # Skip processing if result GLB already exists
            glb_pattern = os.path.join(output_path, f"{base_name}_*sample.glb")
            existing_glbs = glob.glob(glb_pattern)
            if existing_glbs:
                print(f"Skipping frame {base_name} because result GLB already exists: {existing_glbs[0]}")
                continue

            "======================= 3D Mesh Generation per video frame ======================="
            print("/n/n ============= 3D Mesh Generation for frame: ", base_name, " =============/n")
            # Run the pipeline
            if args.model == "TRELLIS":
                rmbg_image, outputs, slat, coords, cond = pipeline.run(
                    image,
                    # Optional parameters
                    seed=seed,
                    save_path=output_path + "/" + base_name + "_rmbg.png",
                )
            elif args.model == "Hunyuan":
                save_path = output_path + "/" + base_name + "_rmbg.png"
                cropped_image, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_rgba=True)
                rmbg_image.save(save_path)
                cropped_image.save(save_path.replace(".png", "_cropped.png"))

                torch.manual_seed(seed)
                try:
                    mesh = pipeline(image=cropped_image)[0]
                except Exception as e:
                    print(f"Error in Hunyuan mesh generation: {e}")
                    continue

                for cleaner in [FloaterRemover(), DegenerateFaceRemover()]:
                    mesh = cleaner(mesh)

                # more facenum, more cost time. The distribution median is ~15000
                mesh = FaceReducer()(mesh, max_facenum=args.max_faces)

                # since in Hunyuan2.0 texture paint, they use xatlas to generate the texture, which may destroy the watertightness. Thus save the attributes of the mesh before painting.
                vertices_watertight = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                faces_watertight = mesh.faces
                mean_point = mesh.vertices.mean(axis=0)
                vertices_watertight = (vertices_watertight - mean_point) * 0.5 + mean_point

                mesh = pipeline_paint(mesh, image=cropped_image)

                # rotate mesh (from y-up to z-up) and scale it to half size to align with the TRELLIS
                mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                mesh.vertices = (mesh.vertices - mean_point) * 0.5 + mean_point

                outputs = {'mesh': [None], 'mesh_genTex': [None]}

                # !NOTICE: The 'mesh.vertex_normals' below may not be accurate since the vertices are customly rotated, not by applying transfrormation matrix in Trimesh.
                '!!!Fingdings!!!: GLB file expects the vertex colors are in linear RGB format, not sRGB format. So if export GLB with vertex color by trimesh, and then import to other viewers like Blender and Windows 3D viewer, the visualized result will be more brighten. (Trimesh itself does not automatically convert the color space, so import again to trimesh will display correctly.)'
                outputs['mesh_genTex'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    # Hunyuan's texture is 2048x2048, so we resize it to 1024x1024
                    res=512, texture=torch.tensor(np.array(mesh.visual.material.image.resize((1024, 1024), Image.Resampling.LANCZOS)), dtype=torch.float32).cuda() / 255, 
                    uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
                )

                # Since hunyaun2.0 texture generation involves xatlas which destroy watertightness, we need to manually calculate the texture
                vertices = vertices_watertight.astype(np.float32)
                faces = faces_watertight.astype(np.int64)
                # bake texture
                observations, extrinsics, intrinsics = render_utils.render_multiview(outputs['mesh_genTex'][0], resolution=1024, nviews=300)
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

                vertices_color = postprocessing_utils.bake_vertice_color(
                    vertices, faces,
                    observations, masks, extrinsics, intrinsics,
                    verbose=True
                )

                mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertices_color)
                outputs['mesh'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    res=512
                )
            elif args.model == "TripoSG" or args.model == "Craftsman":
                save_path = output_path + "/" + base_name + "_rmbg.png"
                _, rmbg_image_rgba, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_all_rbga=True)
                rmbg_image.save(save_path)

                torch.manual_seed(seed)

                if args.model == "TripoSG":
                    vertices, faces, mesh = tripoSG_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_rmbg_net, pipeline_triposg_pipe, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)
                elif args.model == "Craftsman":
                    vertices, faces, mesh = craftsman_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_crafts, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)

                outputs = {'mesh': [None], 'mesh_genTex': [None]}

                outputs['mesh_genTex'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    # Hunyuan's texture is 2048x2048, so we resize it to 1024x1024
                    res=512, texture=torch.tensor(np.array(mesh.visual.material.baseColorTexture.resize((1024, 1024), Image.Resampling.LANCZOS)), dtype=torch.float32).cuda() / 255, 
                    uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
                )

                # bake texture
                observations, extrinsics, intrinsics = render_utils.render_multiview(outputs['mesh_genTex'][0], resolution=1024, nviews=300)
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

                vertices_color = postprocessing_utils.bake_vertice_color(
                    vertices, faces,
                    observations, masks, extrinsics, intrinsics,
                    verbose=True
                )

                mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertices_color)
                outputs['mesh'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    res=512
                )

            # if args.baseline:
            #     if args.model == "TRELLIS":
            #         render_video_and_glbs(outputs, base_name + "_baseline", output_path)
            #         continue
            #     elif args.model == "Hunyuan" or args.model == "TripoSG" or args.model == "Craftsman":
            #         # convert MeshExtractResult to trimesh (use uv and texture)
            #         mesh = outputs['mesh_genTex'][0]
            #         mesh = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), faces=mesh.faces.cpu().numpy(), visual=trimesh.visual.TextureVisuals(uv=mesh.uv.cpu().numpy(), image=Image.fromarray((mesh.texture.cpu().numpy() * 255).astype(np.uint8))), process=False)
            #         mesh.export(output_path + "/" + base_name + "_baseline" + "_sample.glb")
            #         continue

            "============================================================"

            "======== Section 4.1 - Camera Search and Mesh Re-Pose ========"
            print("/n/n ============= Camera Search and Mesh Re-Pose for frame: ", base_name, " =============/n")
            rend_img, params = render_utils.find_closet_camera_pos(outputs['mesh'][0], rmbg_image, save_path=output_path + "/" + base_name, is_Hunyuan=(args.model == "Hunyuan" or args.model == "TripoSG"), use_vggt=args.use_vggt)  # Craftsman is similar to TRELLIS canonical pose
            imageio.imsave(output_path + "/" + base_name + "_sample_mesh_align.png", rend_img, prior_params=params)
            if args.model == "TRELLIS":
                rend_img, params = render_utils.find_closet_camera_pos(outputs['gaussian'][0], rmbg_image, params=params, use_vggt=args.use_vggt)
                imageio.imsave(output_path + "/" + base_name + "_sample_gs_align.png", rend_img)
            else:
                rend_img, params = render_utils.find_closet_camera_pos(outputs['mesh_genTex'][0], rmbg_image, params=params, use_vggt=args.use_vggt)
                imageio.imsave(output_path + "/" + base_name + "_sample_genTex_align.png", rend_img)
            # save camera parameters
            np.save(output_path + "/" + base_name + "_camera_parameters.npy", params)
            # Render the generated mesh from the searched camera position
            yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
            yaw_tensor = torch.tensor([yaw], dtype=torch.float32).cuda()
            pitch_tensor = torch.tensor([pitch], dtype=torch.float32).cuda()
            r_tensor = torch.tensor([r], dtype=torch.float32).cuda()
            lookat_tensor = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()
            
            # Convert camera parameters to extrinsics and intrinsics
            extr, intr = render_utils.optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaw_tensor, pitch_tensor, r_tensor, fov=40, lookat=lookat_tensor
            )
            
            # Render the mesh from the searched camera position
            from v2m4_trellis.renderers import MeshRenderer
            mesh_renderer = MeshRenderer()
            mesh_renderer.rendering_options.resolution = 512
            mesh_renderer.rendering_options.near = 1.0
            mesh_renderer.rendering_options.far = 100.0
            mesh_renderer.rendering_options.ssaa = 4
            
            # Render the mesh (use mesh_genTex if available, otherwise use mesh)
            mesh_to_render = outputs['mesh_genTex'][0] if 'mesh_genTex' in outputs and outputs['mesh_genTex'][0] is not None else outputs['mesh'][0]
            render_result = mesh_renderer.render(mesh_to_render, extr, intr, return_types=["color"])
            
            # Save the rendered image
            rendered_img = np.clip(render_result['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            imageio.imsave(output_path + "/" + base_name + "_rendered_from_searched_camera.png", rendered_img)
            
            "============================================================"

            "======== Section 4.1* - Rendering: Rotate Model and Save ========"
            print("/n/n ============= Rendering: Rotate Model and Save for frame: ", base_name, " =============/n")
            # Rotate the model along its own center for 30 degrees in yaw and save the rendered image
            # Render both with texture and without texture
            for rotation_angle_deg in range(0, 360, 30):

                # Render with texture
                if args.model == "TRELLIS":
                    # For TRELLIS, use gaussian for textured rendering
                    if 'gaussian' in outputs and outputs['gaussian'][0] is not None:
                        mesh_texture = outputs['gaussian'][0]
                        render_rotated_model(mesh_texture, base_name, output_path, params, 
                                            rotation_angle_deg=rotation_angle_deg, suffix="textured",
                                            light_type=args.light_type,
                                            light_location=args.light_location,
                                            light_direction=args.light_direction,
                                            ambient_color=args.ambient_color,
                                            diffuse_color=args.diffuse_color,
                                            specular_color=args.specular_color)
                else:
                    # For other models, use mesh_genTex which has texture
                    if 'mesh_genTex' in outputs and outputs['mesh_genTex'][0] is not None:
                        mesh_texture = outputs['mesh_genTex'][0]
                        render_rotated_model(mesh_texture, base_name, output_path, params, 
                                            rotation_angle_deg=rotation_angle_deg, suffix="textured",
                                            light_type=args.light_type,
                                            light_location=args.light_location,
                                            light_direction=args.light_direction,
                                            ambient_color=args.ambient_color,
                                            diffuse_color=args.diffuse_color,
                                            specular_color=args.specular_color)

                # Render without texture (using mesh without texture)
                mesh_no_texture = get_mesh_without_texture(outputs['mesh'][0])

                render_rotated_model(mesh_no_texture, base_name, output_path, params, 
                                    rotation_angle_deg=rotation_angle_deg, suffix="no_texture", 
                                    light_type=args.light_type,
                                    light_location=args.light_location,
                                    light_direction=args.light_direction,
                                    ambient_color=args.ambient_color,
                                    diffuse_color=args.diffuse_color,
                                    specular_color=args.specular_color)


                # render model aligned with x-axis
                aligned_mesh = align_mesh_bbox_to_x_axis(outputs['mesh'][0])
                render_rotated_model(aligned_mesh, base_name, output_path, params, 
                                    rotation_angle_deg=0, suffix="aligned_with_x_axis", 
                                    light_type=args.light_type,
                                    light_location=args.light_location,
                                    light_direction=args.light_direction,
                                    ambient_color=args.ambient_color,
                                    diffuse_color=args.diffuse_color,
                                    specular_color=args.specular_color)
                render_rotated_model(aligned_mesh, base_name, output_path, params, 
                                    rotation_angle_deg=90, suffix="aligned_with_x_axis_90", 
                                    light_type=args.light_type,
                                    light_location=args.light_location,
                                    light_direction=args.light_direction,
                                    ambient_color=args.ambient_color,
                                    diffuse_color=args.diffuse_color,
                                    specular_color=args.specular_color)
            "============================================================"

            "======== Section 4.2 - Mesh Appearance Refinement via Negative Condition Embedding Optimization (only for TRELLIS)========"
            if args.model == "TRELLIS":
                print("/n/n ============= Mesh Appearance Refinement via Negative Condition Embedding Optimization for frame: ", base_name, " =============/n")
                refer_image = torch.tensor(np.array(rmbg_image)).float().cuda().permute(2, 0, 1) / 255
                start_optimize_iter = 5  # Optimization iteration for the first denoise step T.

                optim_params = {'start_optimize_iter': start_optimize_iter, 'refer_image': refer_image, 'models': pipeline.models, 'normalization': pipeline.slat_normalization, 'camera_params': params}
                # Set the interval as the whole process [0, 1.0] to allow fine-tuning the later steps (which have more details, the default is only [0.5, 1.0]).
                slat = pipeline.sample_slat(cond, coords, {"cfg_interval": [0., 1.0],}, optimize_uncond_noise=optim_params)
                outputs = pipeline.decode_slat(slat, ['mesh', 'gaussian'])

            "============================================================"

            "======== Section 4.1 - Re-canonicalization of the Mesh and Gaussian ========"
            print("/n/n ============= Re-canonicalization of the Mesh and Gaussian for frame: ", base_name, " =============/n")
            # ------ Mesh Part -------
            yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
            yaw = torch.tensor([yaw], dtype=torch.float32).cuda()
            pitch = torch.tensor([pitch], dtype=torch.float32).cuda()
            r = torch.tensor([r], dtype=torch.float32).cuda()
            lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()

            # Get the extrinsics from the camera parameters
            orig = torch.stack([
                torch.sin(yaw) * torch.cos(pitch),
                torch.cos(yaw) * torch.cos(pitch),
                torch.sin(pitch),
            ]).squeeze() * r
            extr = utils3d.torch.extrinsics_look_at(orig, lookat, torch.tensor([0, 0, 1]).float().cuda())
            extr = extr.unsqueeze(0)

            vertices = outputs['mesh'][0].vertices.unsqueeze(0)

            vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
            vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2)).squeeze()

            # Replace the vertices of the mesh (note that there is also Normal that needs transform, but here we temporarily skip that)
            outputs['mesh'][0].vertices = vertices_camera[:, :3]
            
            if args.model == "TRELLIS":
                # ------ Gaussian Part ------
                gs_pos = outputs['gaussian'][0].get_xyz.unsqueeze(0)

                gs_pos_homo = torch.cat([gs_pos, torch.ones_like(gs_pos[..., :1])], dim=-1)
                gs_pos_camera = torch.bmm(gs_pos_homo, extr.transpose(-1, -2)).squeeze()

                outputs['gaussian'][0].from_xyz(gs_pos_camera[:, :3])

                gs_rot = outputs['gaussian'][0].get_rotation

                q_batch = gs_rot / torch.norm(gs_rot, dim=1, keepdim=True)
                # Convert rotation matrix to a single quaternion
                q_matrix = matrix_to_quaternion_batched(extr[0, :3, :3])
                # Perform batched quaternion multiplication
                q_result = reverse_quaternion_multiply_batched(q_matrix, q_batch)
                # Normalize each quaternion in the batch
                q_result = q_result / torch.norm(q_result, dim=1, keepdim=True)

                outputs['gaussian'][0].from_rotation(q_result)

                visual = render_video_and_glbs(outputs, base_name + "_re-canonicalization", output_path, init_extrinsics=extr, inplace_mesh_change=True)

            elif args.model == "Hunyuan" or args.model == "TripoSG" or args.model == "Craftsman":
                mesh_source = outputs['mesh'][0]
                rotated_vertices = mesh_source.vertices.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                faces_np = mesh_source.faces.cpu().numpy()
                vertex_colors = np.clip(mesh_source.vertex_attrs.cpu().numpy()[:, :3] * 255, 0, 255).astype(np.uint8)

                mesh_textured = trimesh.Trimesh(
                    vertices=rotated_vertices,
                    faces=faces_np,
                    vertex_colors=vertex_colors,
                    process=False
                )
                mesh_textured.export(output_path + "/" + base_name + "_re-canonicalization_textured_sample.glb")
                
                mesh_no_tex = trimesh.Trimesh(
                    vertices=rotated_vertices,
                    faces=faces_np,
                    process=False
                )
                mesh_no_tex.export(output_path + "/" + base_name + "_re-canonicalization_no_texture_sample.glb")
                # save obj file
                mesh_no_tex.export(output_path + "/" + base_name + "_re-canonicalization_no_texture_sample.obj")
                # save npy file
                np.save(output_path + "/" + base_name + "_re-canonicalization_no_texture_sample.npy", mesh_no_tex.vertices)