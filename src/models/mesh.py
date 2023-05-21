import kaolin as kal
import torch

import copy

class Mesh:
    def __init__(self,obj_path, device):
        # from https://github.com/threedle/text2mesh

        if ".obj" in obj_path:
            try:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=True, heterogeneous_mesh_handler=kal.io.utils.heterogeneous_mesh_handler_naive_homogenize)
            except:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=False, heterogeneous_mesh_handler=kal.io.utils.heterogeneous_mesh_handler_naive_homogenize)

        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device) #MJ: (2562,3)
        self.faces = mesh.faces.to(device)       #MJ: (5120,3)
        self.normals, self.face_area = self.calculate_face_normals(self.vertices, self.faces)
        self.ft = mesh.face_uvs_idx #MJ: shape =(5120,3); (num_faces,face_size)
        self.vt = mesh.uvs  #MJ: shape =(0,2): no uv coord yet; `(num_uvs, 2)`.

    @staticmethod
    def calculate_face_normals(vertices: torch.Tensor, faces: torch.Tensor):
        """
        calculate per face normals from vertices and faces
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=-1)
        twice_area = torch.norm(n, dim=-1)
        n = n / twice_area[:, None]
        return n, twice_area / 2

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.std(torch.norm(verts, p=2, dim=1))
        verts /= scale
        mesh.vertices = verts
        return mesh

    def normalize_mesh(self,inplace=False, target_scale=1, dy=0):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices  #MJ Size=[3750,3]
        center = verts.mean(dim=0)
        verts = verts - center
        scale = torch.max(torch.norm(verts, p=2, dim=1))
        verts = verts / scale
        verts *= target_scale
        verts[:, 1] += dy  #MJ: verts[:,1] = y coord
        mesh.vertices = verts
        return mesh

