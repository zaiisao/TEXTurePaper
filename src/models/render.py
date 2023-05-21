import kaolin as kal
import torch
import numpy as np
from loguru import logger
class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device) #MJ: np.pi / 3=60
         #MJ: generate_perspective_projection(fov, aspect_ratio)
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim #1200
        self.background = torch.ones(dim).to(device).float() #MJ: self.background= white image

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0): #MJ: look_at_height = 0.25: relative to the world coord system, the mesh origin is assumed to be at the world origin
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)
       
        pos = torch.tensor([x, y, z]).unsqueeze(0) 
        #MJ: the first camera position at (phi,theta)=(0, 1.0472(60deg));  # pos:(3,) => (1,3)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height #MJ: set the y coord (height) of the camera center.
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0) #MJ: direction=camera up dir

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_proj


    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, 'depth map should be negative'
        object_mask = depth_map != 0
        # depth_map[object_mask] = (depth_map[object_mask] - depth_map[object_mask].min()) / (
        #             depth_map[object_mask].max() - depth_map[object_mask].min())
        # depth_map = depth_map ** 4
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map[depth_map == 1] = 0 # Background gets largest value, set to 0

        return depth_map
    #MJ: called by pred_back, _, _ = self.renderer.render_single_view(self.env_sphere,
                                                                #    background_sphere_colors,
                                                                #    elev=theta,
                                                                #    azim=phi,
                                                                #    radius=radius,
                                                                #    dims=dims,
                                                                #    look_at_height=self.dy, calc_depth=False)
    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, radius=2, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        #MJ:  face_vertices_camera[:, :, :, -1] =  The depth value (face_vertices_z): 
        # The depth value (face_vertices_z) is a separate INPUT that represents the depth (z-coordinate)
        # values of the face vertices in camera coordinates. It specifies the distance of each vertex 
        # from the camera viewpoint. The depth values are used to determine the visibility 
        # and ordering of the faces during the rasterization process.
        
        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes) #MJ face_attributes: shape=(1,5120,3,3)
       #MJ:When the rendered face index (face_idx) is equal to -1 for some pixels in the output, it typically indicates that those pixels do not correspond to any valid face. 
       # In other words, those pixels are not within the boundaries of any of the faces being rasterized.
       # In the context of this code, the valid_faces input is a mask of faces being rasterized. By default, if the valid_faces mask is not provided, all faces are considered valid. However, if specific faces are marked as invalid in the valid_faces mask, 
       # the corresponding pixels associated with those faces will have a face_idx value of -1 in the output.
       
       #MJ:  Pixels with face_idx = -1 can have various depth values, 
       # including values greater than 0 or even negative values, depending on the scene and camera setup.
       #MJ: The depth value for a pixel is determined by the scene's geometry and the positioning of objects
       # in relation to the camera viewpoint.
       
        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None): #MJ: background_type="white"
        dims = self.dim if dims is None else dims #MJ: dims=1024

        if render_cache is None: 
            #MJ: render_cache is used only for efficiency, that is to avoid the computation of the same info twice;
            #  It is OK to compute twice the 4 contents of render_cache: depth_map, uv_features, face_normals, face_idx,

            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device)
            
            #MJ: prepare_vertices: 
            # input args: 
            # vertices: the meshes vertices, of shape(batch_size,num_vertices, 3).
            #  faces:             the meshes faces, of shape `(\text{num_faces}, \text{face_size})`.
            #  camera_proj:  the camera projection vector, of shape `(3, 1)`.
            #  camera_rot:  the camera rotation matrices,of shape `(\text{batch_size}, 3, 3)`.
            # camera_trans:  the camera translation vectors, of  shape `(\text{batch_size}, 3)`.            
            # camera_transform: the camera transformation matrices,  of (\text{batch_size}, 4, 3).
            # Replace `camera_trans` and `camera_rot`. 
            # output args:
            # (1) The vertices in camera coordinate indexed by faces,  of shape (batch_size, num_faces, face_size, 3).
            # (2) The vertices in camera plan coordinate indexed by faces,of shape (batch_size,num_faces, face_size, 2).
            # (3) The face normals, of shape (batch_size, num_faces, 3).
            
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)
            # MJ: kal.render.mesh.rasterize() computes the interpolated_features from features (per-vertex per-face) to be drawn, of shape (batch_size,num_faces,3, feature_dim)`
            # MJ: = face_vertices_z, 3D points depth (z) of the face vertices in camera coordinate, of shape (batch_size,num_faces,3)
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1], 
                                                           face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map)

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr) # JA: https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.rasterize
            uv_features = uv_features.detach() # JA: uv_features is the uv coordinates of shape (1,1024,1024,2]); uv_face_attr: shape=(1,7500,3,2)

        else:
            # logger.info('Using render cache')
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map']
        
        mask = (face_idx > -1).float()[..., None] # JA: face_idx = -1 means there is no rendered face corresponding to the pixel. If mask is 1, it means the pixel corresponds to the mesh object.

        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode) 
        # MJ: For each uv coord in the rendered image, get  the texel (R,G,B) at the surface point which corresponds to the uv coord.
        image_features = image_features * mask
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        normals_image = face_normals[0][face_idx, :] # JA: face_idx ranges from -1 to 7499. face_normals.shape = [1, 7500, 3]

        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map}

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache

    def project_uv_single_view(self, verts, faces, uv_face_attr, elev=0, azim=0, radius=2,
                               look_at_height=0.0, dims=None, background_type='none'):
        # project the vertices and interpolate the uv coordinates

        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                          face_vertices_image, uv_face_attr)
        return face_vertices_image, face_vertices_camera, uv_features, face_idx

    def project_single_view(self, verts, faces, elev=0, azim=0, radius=2,
                               look_at_height=0.0):
        # only project the vertices
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        return face_vertices_image
