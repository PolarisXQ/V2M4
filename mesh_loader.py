import os
import sys
sys.path.append("/public/home/xiaqi2025/video/V2M4")

import numpy as np
import trimesh
import torch
import math
import copy
import imageio
import utils3d
import torch.nn.functional as F
from v2m4_trellis.utils.render_utils import render_frames
from v2m4_trellis.representations.mesh.cube2mesh import MeshExtractResult

class RenderConfig:
    def __init__(self, rotation_angle_deg=0, texture=True, align_to_x_axis=False,
                light_type='none', light_location=None, light_direction=None,
                ambient_color=None, diffuse_color=None, specular_color=None):
        """
        Args:
            rotation_angle_deg: The rotation angle in degrees, applied after aligning the mesh to the x axis
            texture: Whether to use texture
            align_to_x_axis: Whether to align the mesh to the x axis
            light_type: The type of light, one of 'point', 'directional', 'none'
            light_location: The location of the light, one of 'up', 'front', 'back', 'left', 'right'
            light_direction: The direction of the light, one of 'up', 'front', 'back', 'left', 'right'
            ambient_color: The ambient color of the light, a list of 3 floats
            diffuse_color: The diffuse color of the light, a list of 3 floats
            specular_color: The specular color of the light, a list of 3 floats
        """
        self.rotation_angle_deg = rotation_angle_deg
        self.texture = texture
        self.align_to_x_axis = align_to_x_axis
        self.light_type = light_type
        self.light_location = light_location
        self.light_direction = light_direction
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color

class MeshLoader:
    def __init__(self, data_root):
        self.data_root = data_root

    def load_mesh(self, major_class_name:str, frame_index:str):
        # check if major_class_name is in the data_root
        if major_class_name not in os.listdir(self.data_root):
            raise ValueError(f"Major class name {major_class_name} not found in {self.data_root}")
        if not os.path.exists(os.path.join(self.data_root, major_class_name, f"{frame_index}_re-canonicalization_textured_sample.glb")):
            raise ValueError(f"Frame index {frame_index} not found in {os.path.join(self.data_root, major_class_name)}")
        mesh_path = os.path.join(self.data_root, major_class_name, f"{frame_index}_re-canonicalization_textured_sample.glb")

        tm_mesh = trimesh.load(mesh_path, process=False, force='mesh')

        if isinstance(tm_mesh, trimesh.Scene):
            if len(tm_mesh.geometry) == 0:
                raise ValueError(f"No geometry found in mesh file {mesh_path}")
            tm_mesh = tm_mesh.as_mesh()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vertices_np = tm_mesh.vertices.astype(np.float32)
        faces_np = tm_mesh.faces.astype(np.int64)

        vertices = torch.from_numpy(vertices_np).to(device)
        faces = torch.from_numpy(faces_np).long().to(device)

        colors = None
        if hasattr(tm_mesh, "visual") and getattr(tm_mesh.visual, "vertex_colors", None) is not None and len(tm_mesh.visual.vertex_colors):
            colors_np = np.asarray(tm_mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
            colors = torch.from_numpy(colors_np).to(device)
        else:
            colors = torch.full((vertices.shape[0], 3), 0.7, dtype=torch.float32, device=device)

        vertex_attrs = torch.cat([colors, torch.zeros_like(colors)], dim=1)

        mesh_result = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=vertex_attrs)
        normals = mesh_result.comput_v_normals(vertices, faces)
        if normals is not None:
            mesh_result.vertex_attrs[:, 3:6] = normals

        return mesh_result

    def load_camera_parameters(self, major_class_name, frame_index):
        camera_parameters_path = os.path.join(self.data_root, major_class_name, f"{frame_index}_camera_parameters.npy")
        return np.load(camera_parameters_path)

    def get_mesh_without_texture(self, mesh):
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


    def align_mesh_bbox_to_x_axis(self, mesh):
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

    def apply_lighting_nvdiffrast(self, target_mesh, params, ambient_color=[0.2, 0.2, 0.2], diffuse_color=[1.0, 1.0, 1.0], specular_color=[0.2, 0.2, 0.2], light_type='point', light_location=None, light_direction=None):
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

    def rotate_render_and_save(self, mesh, base_name, output_path, params, rotation_angle_deg=30, suffix="", 
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
        
        render_options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

        
        result = render_frames(rotated_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
    
        # Save the rendered image
        suffix_str = f"_{suffix}" if suffix else ""
        output_filename = output_path + "/" + base_name + f"_rotated_{rotation_angle_deg}deg{suffix_str}_no_light.png"
        imageio.imsave(output_filename, rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='up')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_up_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='front')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_front_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='back')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_back_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='left')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_left_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location='right')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_right_point_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='up')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_up_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='front')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_front_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='back')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_back_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='left')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_left_directional_light.png'), rend_img)

        lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction='right')
        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        imageio.imsave(output_filename.replace('_no_light.png', '_right_directional_light.png'), rend_img)

    def render(self, mesh, params, rotation_angle_deg, light_type, light_location, light_direction, ambient_color, diffuse_color, specular_color):
        """
        Render the model with the given rotation angle, light type, light location, light direction, ambient color, diffuse color, specular color.

        Args:
            mesh: The mesh to render
            params: The camera parameters
            rotation_angle_deg: The rotation angle in degrees
            light_type: The type of light
            light_location: The location of the light
            light_direction: The direction of the light
            ambient_color: The ambient color
            diffuse_color: The diffuse color
            specular_color: The specular color

        Returns:
            rend_img: rendered image
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

        render_options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

        if light_type == 'point':
            lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='point', light_location=light_location)
        elif light_type == 'directional':
            lighted_mesh = self.apply_lighting_nvdiffrast(rotated_mesh, params, light_type='directional', light_direction=light_direction)
        elif light_type == 'none':
            lighted_mesh = rotated_mesh
        else:
            raise ValueError(f"Invalid light type: {light_type}")

        result = render_frames(lighted_mesh, extr, intr_batch, render_options)
        rend_img = result['color'][0]
        return rend_img

    def load_mesh_and_render(self, major_class_name, frame_index, render_config:list[RenderConfig]):
        """
        Load the mesh and render the model with the given render configuration.

        Args:
            major_class_name: The name of the major class
            frame_index: The index of the frame, selected from 'head', 'tail', 'middle' (recommended) or a 5 digits with leading zeros string
            render_config: The configuration for the render, a list of RenderConfig objects
        
        Returns:
            len(render_config) rendered images, each image is a numpy array of shape (512, 512, 3)
        """
        # cope with frame_index
        if frame_index in ["head", "tail", "middle"]:
            # get availale frame indices
            # find file end with _re-canonicalization_textured_sample.glb
            glb_files = [f for f in os.listdir(os.path.join(self.data_root, major_class_name)) if f.endswith("_re-canonicalization_textured_sample.glb")]
            frame_indices = [f.split("_")[0] for f in glb_files]
            frame_indices = [int(f) for f in frame_indices]
            frame_indices.sort()
            if frame_index == "head":
                frame_index = frame_indices[0]
            elif frame_index == "tail":
                frame_index = frame_indices[-1]
            else:
                frame_index = frame_indices[1]
            # format to 5 digits
            frame_index = str(frame_index).zfill(5)

        mesh = self.load_mesh(major_class_name, frame_index)
        params = self.load_camera_parameters(major_class_name, frame_index)
        rendered_images = []
        for config in render_config:
            if config.align_to_x_axis:
                mesh = self.align_mesh_bbox_to_x_axis(mesh)
            if not config.texture:
                mesh = self.get_mesh_without_texture(mesh)
            rendered_image = self.render(mesh, params, config.rotation_angle_deg, config.light_type, config.light_location, config.light_direction, config.ambient_color, config.diffuse_color, config.specular_color)
            rendered_images.append(rendered_image)
        return rendered_images

if __name__ == "__main__":
    mesh_loader = MeshLoader(data_root="/inspurfs/group/mayuexin/xiaqi/DAVIS-2017/V2M4_results")
    render_configs = [RenderConfig(rotation_angle_deg=0, texture=True, align_to_x_axis=True,
                                  light_type='none'),
                                  RenderConfig(rotation_angle_deg=90, texture=True, align_to_x_axis=True,
                                  light_type='directional', light_direction='front'),
                                  RenderConfig(rotation_angle_deg=180, texture=True, align_to_x_axis=True,
                                  light_type='directional', light_direction='back'),
                                  RenderConfig(rotation_angle_deg=270, texture=True, align_to_x_axis=True,
                                  light_type='directional', light_direction='left'),
                                  RenderConfig(rotation_angle_deg=60, texture=True, align_to_x_axis=False,
                                  light_type='none') ,
                                  RenderConfig(rotation_angle_deg=-60, texture=True, align_to_x_axis=False,
                                  light_type='none') ,
                                  RenderConfig(rotation_angle_deg=120, texture=False, align_to_x_axis=False,
                                  light_type='point', light_location='front'),
                                  RenderConfig(rotation_angle_deg=-120, texture=False, align_to_x_axis=False,
                                  light_type='point', light_location='front')]
    rendered_images = mesh_loader.load_mesh_and_render("bear", "head", render_configs)
    os.makedirs("bear_renders", exist_ok=True)
    for i, rendered_image in enumerate(rendered_images):
        texture_type = "textured" if render_configs[i].texture else "untextured"    
        align_type = "aligned" if render_configs[i].align_to_x_axis else "unaligned"
        light_type = render_configs[i].light_type
        light_location = render_configs[i].light_location
        light_direction = render_configs[i].light_direction
        image_name = os.path.join("bear_renders", f"bear_{render_configs[i].rotation_angle_deg}deg_{texture_type}_{align_type}_{light_type}_{light_location}_{light_direction}.png")
        imageio.imsave(image_name, rendered_image)