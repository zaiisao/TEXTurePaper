import os

import kaolin as kal
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from .mesh import Mesh
from .render import Renderer
from src.configs.train_config import GuideConfig


def build_cotan_laplacian_torch(points_tensor: torch.Tensor, tris_tensor: torch.Tensor) -> np.ndarray:
    tris, points = tris_tensor.cpu().numpy(), points_tensor.cpu().numpy()

    a, b, c = (tris[:, 0], tris[:, 1], tris[:, 2])
    A = np.take(points, a, axis=1)
    B = np.take(points, b, axis=1)
    C = np.take(points, c, axis=1)

    eab, ebc, eca = (B - A, C - B, A - C)
    eab = eab / np.linalg.norm(eab, axis=0)[None, :]
    ebc = ebc / np.linalg.norm(ebc, axis=0)[None, :]
    eca = eca / np.linalg.norm(eca, axis=0)[None, :]

    alpha = np.arccos(-np.sum(eca * eab, axis=0))
    beta = np.arccos(-np.sum(eab * ebc, axis=0))
    gamma = np.arccos(-np.sum(ebc * eca, axis=0))

    wab, wbc, wca = (1.0 / np.tan(gamma), 1.0 / np.tan(alpha), 1.0 / np.tan(beta))
    rows = np.concatenate((a, b, a, b, b, c, b, c, c, a, c, a), axis=0)
    cols = np.concatenate((a, b, b, a, b, c, c, b, c, a, a, c), axis=0)
    vals = np.concatenate((wab, wab, -wab, -wab, wbc, wbc, -wbc, -wbc, wca, wca, -wca, -wca), axis=0)
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(points.shape[1], points.shape[1]), dtype=float).tocsc()
    return L


def build_graph_laplacian_torch(tris_tensor: torch.Tensor) -> np.ndarray:
    tris = tris_tensor.cpu().numpy()
    n_verts = tris.max() + 1
    v2v = [[] for _ in range(n_verts)]
    for face in tris:
        for i in face:
            for j in face:
                if i != j:
                    if j not in v2v[i]:
                        v2v[i].append(j)

    valency = [len(x) for x in v2v]
    I, J, vals = [], [], []
    for i in range(n_verts):
        I.append(i)
        J.append(i)
        vals.append(1)

        for neighbor in v2v[i]:
            I.append(i)
            J.append(neighbor)
            vals.append(-1 / valency[i])
    L = sparse.csr_matrix((vals, (I, J)), shape=(n_verts, n_verts))
    return L


def eigen_problem(Lap, k=20, e=0.0) -> (torch.Tensor, torch.Tensor):
    shift = 1e-4
    eigenvalues, eigenvectors = eigsh(
        Lap + shift * scipy.sparse.eye(Lap.shape[0]),
        k=k + 1, which='LM', sigma=e, tol=1e-3)
    eigenvalues += shift

    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    return torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors.T).float()


def choose_multi_modal(n: int, k: int):
    interval_length = n // k
    n_intervals = n / interval_length
    if n_intervals == int(n_intervals):
        n_intervals = int(n_intervals)
    else:
        n_intervals = int(n_intervals) + 1
    chosen_numbers = []
    for i in range(n_intervals):
        current_interval_length = min(interval_length, n - i * interval_length)
        chosen_numbers.append(np.random.choice(current_interval_length) + i * interval_length)
    return chosen_numbers


class TexturedMeshModel(nn.Module):
    def __init__(self,
                 opt: GuideConfig,
                 render_grid_size=1024,
                 texture_resolution=1024,
                 initial_texture_path=None,
                 cache_path=None,
                 device=torch.device('cpu'),
                 augmentations=False,
                 augment_prob=0.5):

        super().__init__()
        self.device = device
        self.augmentations = augmentations
        self.augment_prob = augment_prob
        self.opt = opt
        self.dy = self.opt.dy                  #MJ: dy = 0.25 m = look_at height of the camera
        self.mesh_scale = self.opt.shape_scale #MJ: 0.6
        self.texture_resolution = texture_resolution #MJ: 1024
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = self.opt.initial_texture #MJ: None
        self.cache_path = cache_path
        self.num_features = 3

        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size),
                                 interpolation_mode=self.opt.texture_interpolation_mode)
        self.env_sphere, self.mesh = self.init_meshes()
        self.default_color = [0.8, 0.1, 0.8]  #MJ: default_color = purple color?
        self.background_sphere_colors, self.texture_img = self.init_paint() #MJ: create the background color Parameters and the texture color Parameters
        #breakpoint()
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        if self.opt.reference_texture: #=None
            base_texture = torch.Tensor(np.array(Image.open(self.opt.reference_texture).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
            change_mask = (
                    (base_texture.to(self.device) - self.texture_img).abs().sum(axis=1) > 0.1).float()
            with torch.no_grad():
                self.meta_texture_img[:, 1] = change_mask
        self.vt, self.ft = self.init_texture_map()  #MJ: self.vt is the vertex features (uvs) of shape (4839,2);self.ft=face_uvs_idx of shape (num_faces,face_size)=(7500,3)
        
        #MJ:In summary, the function aims to perform the indexing and rearrangement of the vertex features to obtain
        # a tensor that represents features per vertex per face, 
        # allowing for subsequent computations or analysis specific to each vertex in relation to its associated face in a mesh.

        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0), #MJ: shape = (batch_size}, num_points, knum) =(1,4839,2)
            self.ft.long()        #MJ: shape = num_faces,face_size)=(7500,3)
            ).detach()
        # self.face_attributes: (batch_size, num_faces, num_vertices, knum)

        self.n_eigen_values = 20
        self._L = None
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def L(self) -> np.ndarray:
        if self._L is None:
            self._L = build_cotan_laplacian_torch(self.mesh.vertices.T, self.mesh.faces)
        return self._L

    def eigens(self, k: int, e: float) -> (torch.Tensor, torch.Tensor):
        if self._eigenvalues is None or self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = eigen_problem(self.L, k, e)
            self._eigenvalues, self._eigenvectors = \
                self._eigenvalues.to(self.device), self._eigenvectors.to(self.device)

        return self._eigenvalues, self._eigenvectors

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
        vertices -= vertices.mean(dim=0)[None, :]
        vertices /= vertices.norm(dim=1).max()
        vertices *= mesh_scale
        vertices[:, 1] += dy
        return vertices

    def spectral_augmentations(self, vertices: torch.Tensor) -> torch.Tensor:
        eigen_values, basis_functions = self.eigens(self.n_eigen_values, 0.0)
        basis_functions /= basis_functions.max(dim=-1)[0][:, None] - basis_functions.min(dim=-1)[0][:, None]

        chosen_basis_function = choose_multi_modal(basis_functions.shape[0], 2)
        coeffs = torch.zeros(basis_functions.shape[0]).to(self.device)
        signs = ((torch.rand(len(chosen_basis_function), device=self.device) > 0.5).float() - 0.5) * 2
        coeffs[chosen_basis_function] = signs

        reconstructed = coeffs @ basis_functions.float()

        directions = vertices / torch.norm(vertices, dim=1)[:, None]
        deformed_v = vertices + 0.25 * reconstructed[:, None] * directions
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def axis_augmentations(self, vertices: torch.Tensor, stretch_factor: float = 1.6, squish_factor: float = 0.7):
        axis_indices = np.arange(0, 3)
        axis_indices = np.random.permutation(axis_indices)
        stretch_axis = axis_indices[0]
        squish_axis = axis_indices[1]

        deformed_v = vertices.clone()
        deformed_v[:, stretch_axis] *= stretch_factor
        deformed_v[:, squish_axis] *= squish_factor
        return self.normalize_vertices(deformed_v, mesh_scale=self.mesh_scale, dy=self.dy)

    def augment_vertices(self):
        verts = self.mesh.vertices.clone()
        if np.random.rand() < 0.5:
            verts = self.spectral_augmentations(verts)
        if np.random.rand() < 0.5:
            verts = self.axis_augmentations(verts)
        return verts

    def init_meshes(self, env_sphere_path='shapes/env_sphere.obj'):
        # JA: When this function is called, env_sphere_path is not defined so it is set to the default path written here.
        # However, no such file under the name of env_sphere.obj seems to exist.
        env_sphere = Mesh(env_sphere_path, self.device)

        mesh = Mesh(self.opt.shape_path, self.device)
        mesh.normalize_mesh(inplace=True, target_scale=self.mesh_scale, dy=self.dy)

        return env_sphere, mesh

    def zero_meta(self):
        with torch.no_grad():
            self.meta_texture_img[:] = 0

    def init_paint(self, num_backgrounds=1):
        # random color face attributes for background sphere
        init_background_bases = torch.rand(num_backgrounds, 3).to(self.device) #get three uniform random numbers of shape (1,3) => random color
        modulated_init_background_bases_latent = init_background_bases[:, None, None, :] * 0.8 + 0.2 * torch.randn(
            num_backgrounds, self.env_sphere.faces.shape[0], # self.env_sphere.faces.shape[0] =5120
            3, self.num_features, dtype=torch.float32).cuda() # self.num_features (num of vertices on a face)=3; (1,5120,3,3) + (1,1,1,3)
        #breakpoint()=3
        #breakpoint()
        background_sphere_colors = nn.Parameter(modulated_init_background_bases_latent.cuda()) #MJ: colors for every vertex of every face of the background mesh

        if self.initial_texture_path is not None:
            texture = torch.Tensor(np.array(Image.open(self.initial_texture_path).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
        else: #self.initial_texture_path=None: assign the default color (purple) to every pixel of the texture
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).cuda() * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).cuda() #MJ: self.texture_resolution=1024; image dim = 1200
        #breakpoint()
        texture_img = nn.Parameter(texture)
        return background_sphere_colors, texture_img

    def invert_color(self, color: torch.Tensor) -> torch.Tensor:
        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2

        pinv = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T)
        if len(color) == 1 or type(color) is torch.Tensor:
            init_color_in_latent = color @ pinv.T
        else:
            init_color_in_latent = pinv @ torch.tensor(
                list(color)).float().to(A.device)
        return init_color_in_latent

    def change_default_to_median(self):
        diff = (self.texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        default_mask = (diff < 0.1).float().unsqueeze(0)
        median_color = self.texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(axis=1)
        with torch.no_grad():
            self.texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)

    def init_texture_map(self):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        if self.mesh.vt is not None and self.mesh.ft is not None \
                and self.mesh.vt.shape[0] > 0 and self.mesh.ft.min() > -1:
            vt = self.mesh.vt.cuda()
            ft = self.mesh.ft.cuda()
            run_xatlas = not (vt.shape[0] == self.mesh.vertices.shape[0] and ft.shape[0] == self.mesh.faces.shape[0])
        elif cache_exists_flag:
            vt = torch.load(vt_cache).cuda()  #MJ:=(4839,2)
            ft = torch.load(ft_cache).cuda()  #=(7500,3)
            run_xatlas = not (vt.shape[0] == self.mesh.vertices.shape[0] and ft.shape[0] == self.mesh.faces.shape[0])
        else:
            run_xatlas = True

        if run_xatlas:
            # unwrap uvs: 
            # UV unwrapping refers to the process of creating a 2D representation of a 3D surface's texture coordinates. 
            # It involves flattening the surface onto a 2D plane,
            # allowing texture artists to paint or apply textures accurately on the surface.
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()  #= (3750,3)
            f_np = self.mesh.faces.int().cpu().numpy() #=(7500,3)
            logger.info(f'running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options) #=>  This generates the UV coordinates for the mesh.
            #MJ: Extract the UV coordinates, face indices, and vertex indices from the atlas  
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]  vt_np: (4839,2): there are 4389 vertices and each vertex has 2 UV coordinates

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()  #=(4839,2) ?? what is it? => uvs of the vertices
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()      #(7500,3)
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                torch.save(vt.cpu(), vt_cache)
                torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return [self.background_sphere_colors, self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_mesh(self, path):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        h0, w0 = 256, 256
        ssaa, name = 1, ''

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        colors = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)

        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        colors = Image.fromarray(colors)

        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)

        colors.save(os.path.join(path, f'{name}albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')

        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info('writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {v[0]} {v[1]} \n')

            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    # JA: Here self is the TexturedMeshModel
    def render(self, theta=None, phi=None, radius=None, background=None, #MJ: background is Not None
               use_meta_texture=False, render_cache=None, use_median=False, dims=None):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        background_sphere_colors = self.background_sphere_colors[
            torch.randint(0, self.background_sphere_colors.shape[0], (1,))] # JA: background_sphere_colors.shape: [1, 5120, 3, 3]
        # torch.randint(0, self.background_sphere_colors.shape[0], (1,)) is always [0],
        # with self.background_sphere_colors.shape[0]=1; background_sphere_colors: shape=(5120,3,3)
        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img

        if self.augmentations:
            augmented_vertices = self.augment_vertices()
        else:
            augmented_vertices = self.mesh.vertices

        #MJ: The following code changes the pixels of texture_img that are close to the default color (purple color) to the average color of the pixels that are NOT close to the default color. 
        #Note: The same logic that uses the notion of the default color is also used within calculate_trimap():
        # def calculate_trimap(self, rgb_render_raw: torch.Tensor,
        #                  depth_render: torch.Tensor,
        #                  z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
        #                  mask: torch.Tensor):
        
        #   diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
        #     self.device)).abs().sum(axis=1)
        #   exact_generate_mask = (diff < 0.1).float().unsqueeze(0) # JA: exact_generate_mask.shape = [1, 1, 1200, 1200] why???



        if use_median: #MJ: true when self.paint_step > 1
            #MJ: You could have called  change_default_to_median(self) instead of the following block
            
            diff = (texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)
            default_mask = (diff < 0.1).float().unsqueeze(0) #MJ: diff: shape=(1,height, width); the 1 pixels
            # represents pixels close to the default color, purple color. default_mask: shape=(1,1,H,W)
            median_color = texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)  #MJ: median_color is the average of the pixel colors that are NOT close to the default purple color
            texture_img = texture_img.clone() #MJ:  texture_img[0, :].reshape(3, -1): shape=(3,H*W)
            with torch.no_grad(): #MJ: median_color: shape=(3,); 
                # set texture_img so that the pixels close to the default color become the median color 
                texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str:
            background_type = background
            use_render_back = True
        # JA: mask is the image region where there are face indices
        # pred_features is the interpolated texture map (R,G,B)'s: image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode) 
        pred_features, mask, depth, normals, render_cache = self.renderer.render_single_view_texture(augmented_vertices,
                                                                                                     self.mesh.faces,
                                                                                                     self.face_attributes, #(1,7500,3,2)
                                                                                                     texture_img,
                                                                                                     elev=theta,
                                                                                                     azim=phi,
                                                                                                     radius=radius,
                                                                                                     look_at_height=self.dy,
                                                                                                     render_cache=render_cache,
                                                                                                     dims=dims,
                                                                                                     background_type=background_type) #MJ: background_type='none'

        mask = mask.detach()

        #MJ: Get the predicted background
        if use_render_back: # JA: Render the background
            pred_map = pred_features
            pred_back = pred_features
        else: # JA: pred_back is defined differently
            if background is None: #MJ: use env_sphere and background_sphere_colors to get the background
                pred_back, _, _ = self.renderer.render_single_view(self.env_sphere,
                                                                   background_sphere_colors, #(1,5120,3,3)
                                                                   elev=theta,
                                                                   azim=phi,
                                                                   radius=radius,
                                                                   dims=dims,
                                                                   look_at_height=self.dy, calc_depth=False)
            elif len(background.shape) == 1:
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:  #MJ: this is the case: 
                pred_back = background
            # MJ: Combine the predicted features and the predicted background to get pred_map
            pred_map = pred_back * (1 - mask) + pred_features * mask

        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)

        return {'image': pred_map, 'mask': mask, 'background': pred_back,
                'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
                'texture_map': texture_img} # JA: texture_map is either self.texture_img or self.meta_texture_img (which is the global data)

    def draw(self, theta, phi, radius, target_rgb):
        # failed attempt to draw on the texture image

        uv_features, face_idx = self.renderer.project_uv_single_view(self.mesh.vertices,
                                                                     self.mesh.faces,
                                                                     self.face_attributes,
                                                                     elev=theta,
                                                                     azim=phi,
                                                                     radius=radius,
                                                                     look_at_height=self.dy)
        unique_face_idx = torch.unique(face_idx)
        print('')
