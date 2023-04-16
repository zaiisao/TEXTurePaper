import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs.train_config import RenderConfig
from src.utils import get_view_direction
from loguru import logger

# JA: NOT USED
def rand_poses(size, device, radius_range=(1.0, 1.5), theta_range=(0.0, 150.0), phi_range=(0.0, 360.0),
               angle_overhead=30.0, angle_front=60.0, biased_angles=True):
    if theta_range != (0.0, 180.0):
        warnings.warn("theta_range is not (0.0, 180.0) in rand_poses\n Will use (0.0, 180.0) instead")

    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)

    if biased_angles:
        top_flag = np.random.rand() > 0.3  # 70% of the time, the camera is at the top
        if top_flag:
            x = 1 - torch.rand(size, device=device)
            thetas = torch.acos(x)
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        else:
            x = 1 - (torch.rand(size, device=device) + 1)
            thetas = torch.acos(x)
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    else:
        # logger.warning('Using old theta calc')
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # thetas = torch.acos(1-2*torch.rand(size, device=device))

        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius.item()

# JA: NOT USED
def rand_modal_poses(size, device, radius_range=(1.4, 1.6), theta_range=(45.0, 90.0), phi_range=(0.0, 360.0),
                     angle_overhead=30.0, theta_range_overhead=(0.0, 20.0), angle_front=60.0):
    theta_range = np.deg2rad(theta_range)
    theta_range_overhead = np.deg2rad(theta_range_overhead)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    overhead_flag = torch.rand(1, device=device) > 0.85
    if overhead_flag:
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = torch.rand(size, device=device) * (theta_range_overhead[1] - theta_range_overhead[0]) + \
                 theta_range_overhead[0]
    else:
        phi_mods = np.deg2rad([0, 90, 180, 270])
        pertube_magnitude = np.deg2rad(15)
        rand_pertubations = torch.rand(size, device=device) * pertube_magnitude
        phis = rand_pertubations + torch.from_numpy(phi_mods[np.random.randint(0, 4, size)]).to(device)
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius.item()


# JA: Even though angle_overhead and angle_front have default values, every
# function call uses the self.cfg.overhead_range and self.cfg.front_range
def circle_poses(device, radius=1.25, theta=60.0, phi=0.0, angle_overhead=30.0, angle_front=60.0):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    # JA: Given the theta (camera elevation) and phi (azimuth camera angle),
    # dirs is set to the enum value corresponding to view direction.
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    #
    # Although the variables are named plural (thetas, phis), each is simply
    # a tensor with one float.
    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius

# JA: This dataset class is used for the training dataset.
class MultiviewDataset:
    def __init__(self, cfg: RenderConfig, device):
        super().__init__()

        #breakpoint()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, tests

        # JA: From train_config.py: Number of views to use (It is currently set as 8)
        size = self.cfg.n_views

        # JA: phi values are in degrees, and the resulting values should be: [0, 45, 90, 135, 180, 225, 270, 315]
        self.phis = [(index / size) * 360 for index in range(size)]
        self.thetas = [self.cfg.base_theta for _ in range(size)] # JA: [60, 60, 60, 60, 60, 60, 60, 60]

        # JA: Reference on "lambda" usage and syntax: https://wikidocs.net/64
        # The 0th element of alternate_lists remains the 0th element after transformation no matter what.

        # Alternate lists
        alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2], l[-1:size // 2:-1]) for i in j] + [
            l[size // 2]]
        if self.cfg.alternate_views:
            self.phis = alternate_lists(self.phis)  # JA: [0, 45, 90, 135, 180, 225, 270, 315] -> [0, 45, 315, 90, 270, 135, 225, 180]
                                                    # The 0th element of alternate_lists always remains the 0th element after transformation.
                                                    # For every element after, repeat until there are no more "remaining elements":
                                                    #   The first remaining element of the original list is placed as the next element of the new list.
                                                    #   The last remaining element of the original list is placed as the next element of the new list. 
            self.thetas = alternate_lists(self.thetas) # JA: [60, 60, 60, 60, 60, 60, 60, 60] -> [60, 60, 60, 60, 60, 60, 60, 60]
        logger.info(f'phis: {self.phis}')
        # self.phis = self.phis[1:2]
        # self.thetas = self.thetas[1:2]
        # if append_upper:
        #     # self.phis = [0,180, 0, 180]+self.phis
        #     # self.thetas =[30, 30, 150, 150]+self.thetas
        #     self.phis =[180,180]+self.phis
        #     self.thetas = [30,150]+self.thetas

        # JA: views_before is an empty list by default
        for phi, theta in self.cfg.views_before:
            self.phis = [phi] + self.phis
            self.thetas = [theta] + self.thetas

        # JA: views_after is a list of tuples consisting of [180, 30], [180, 150] by default, formatted as [phi, theta]
        # phi       [0, 45, 315, 90, 270, 135, 225, 180] -> [0, 45, 315, 90, 270, 135, 225, 180, 180, 180]
        # theta     [60, 60, 60, 60, 60, 60, 60, 60]     -> [60, 60, 60, 60, 60, 60, 60, 60, 30, 150]
        for phi, theta in self.cfg.views_after:
            self.phis = self.phis + [phi]
            self.thetas = self.thetas + [theta]
            # self.phis = [0, 0] + self.phis
            # self.thetas = [20, 160] + self.thetas

        self.size = len(self.phis)

    def collate(self, index):

        # B = len(index)  # always 1

        # phi = (index[0] / self.size) * 360

        # JA: len(index) is always 1 because we only use a batch size of 1
        phi = self.phis[index[0]]
        theta = self.thetas[index[0]]
        radius = self.cfg.radius

        
        dirs, thetas, phis, radius = circle_poses(self.device, radius=radius, theta=theta,
                                                  phi=phi,
                                                  angle_overhead=self.cfg.overhead_range,
                                                  angle_front=self.cfg.front_range)

        data = {
            'dir': dirs,
            'theta': thetas,
            'phi': phis,
            'radius': radius
        }

        return data

    def dataloader(self):
        # JA: Here, list(range(self.size)) is the dataset. self.size is simply the length of phis which
        # is the same as the self.cfg.n_views plus the lengths of views_before and views_after.
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #
        # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
        #    batch_sampler=None, num_workers=0, collate_fn=None,
        #    pin_memory=False, drop_last=False, timeout=0,
        #    worker_init_fn=None, *, prefetch_factor=2,
        #    persistent_workers=False)

        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader

# JA: This dataset class is used for the validation dataset.
class ViewsDataset:
    def __init__(self, cfg: RenderConfig, device, size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test
        self.size = size

    def collate(self, index):
        # circle pose
        phi = (index[0] / self.size) * 360
        dirs, thetas, phis, radius = circle_poses(self.device, radius=self.cfg.radius * 1.2, theta=self.cfg.base_theta,
                                                  phi=phi,
                                                  angle_overhead=self.cfg.overhead_range,
                                                  angle_front=self.cfg.front_range)

        data = {
            'dir': dirs,
            'theta': thetas,
            'phi': phis,
            'radius': radius
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader
