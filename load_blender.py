import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from load_llff import *


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, near=0., far=1., bd_factor=0.75, path_zflat=False):

    metas = None
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        metas = json.load(fp)

    meta = metas
    imgs = []
    # imgs_64 = []
    poses = []
    
    # train_img_num = 0
    # viz_img_num = 0

    for i, frame in enumerate(meta['frames']):

        fname = frame['file_path'] + '.png'

        if i >= 34:
            fname = fname.replace('images_1', 'no_RS')

        img = imageio.imread(fname)

        imgs.append(img)
        poses.append(np.array(frame['transform_matrix']))

    imgs = (np.array(imgs)[..., :3] / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    sc = 1. / (near * bd_factor)
    poses[:, :3, 3] *= sc  # T
    near *= sc
    far *= sc

    poses = recenter_poses(poses)

    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near * .9, far * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 50, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
        #             zloc = np.percentile(tt, 10, 0)[2]
        zloc = -close_depth * .1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.
        N_rots = 1
        N_views /= 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    imgs = torch.Tensor(imgs)
    poses = torch.Tensor(poses)
    render_poses = torch.Tensor(render_poses)

    i_val = np.array([i + 34 for i in range(34)]).astype(np.int8)
    i_test = np.array([i + 34 + 34 for i in range(16)]).astype(np.int8)

    i_train = np.array([i for i in range(34)]).astype(np.int8)
        
    return imgs, poses, render_poses, [H, W, focal], near, far, i_train, i_val, i_test

