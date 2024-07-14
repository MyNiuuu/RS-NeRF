import torch
import torch.nn as nn
import torch.nn.functional as F

from run_nerf import *

from Spline import se3_to_SE3


max_iter = 200000
T = max_iter+1
BOUNDARY = 20


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(args, multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False if args.barf else True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Model():
    def __init__(self):
        super().__init__()

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance>0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        return self.optim


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # network
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, barf_i, pts, viewdirs, args):


        embed_fn, input_ch = get_embedder(args, args.multires, args.i_embed)
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args, args.multires_views, args.i_embed)
        
        
        
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embedded = embed_fn(pts_flat)



        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        input_pts, input_views = torch.split(embedded, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        outputs = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])

        return outputs

    def raw2output(self, raw, z_vals, rays_d, raw_noise_std=0.0):

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,:-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        depth_map = torch.sum(weights * z_vals, -1)

        # print(depth_map.shape)

        # assert False
        # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

        disp_map = torch.max(1e-6 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-6))
        acc_map = torch.sum(weights, -1)

        sigma = F.relu(raw[..., 3] + noise)

        return rgb_map, disp_map, acc_map, weights, depth_map, sigma


class Graph(nn.Module):

    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False, special_iter=10000):
        super().__init__()
        self.special_iter = special_iter
        self.nerf = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        if args.N_importance > 0:
            self.nerf_fine = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)

    def forward(self, i, multisample_count, img_idx, row_idx, point_idx, poses_num, H, W, K, args, novel_view=False):
        
        if novel_view:
            assert False
            poses_sharp = se3_to_SE3(self.se3_sharp.weight)
            ray_idx_sharp = torch.randperm(H * W)[:300]
            ret = self.render(i, poses_sharp, ray_idx_sharp, H, W, K, args)
            return ret, ray_idx_sharp, poses_sharp


        if multisample_count is not None:

            multisample_count = multisample_count[img_idx]

            row_idx_expand = row_idx.unsqueeze(1).expand(-1, point_idx.shape[-1])
     
            multisample_ranges = multisample_count[:, row_idx_expand, point_idx]  # [34, 64, 2, 2]

            lower_bounds = -multisample_ranges[..., 0]
            upper_bounds = multisample_ranges[..., 1]

            rand_vals = torch.rand_like(lower_bounds.float()).to(row_idx_expand.device)
            row_disp = lower_bounds + rand_vals * (upper_bounds - lower_bounds)
            row_disp = row_disp.round().long()  # 转换为整数

            row_idx_after = row_idx_expand.clone().unsqueeze(0).repeat(args.pic_num, 1, 1)

            row_idx_after = row_idx_after + row_disp  # [34, 64, 2]
            row_idx_after[row_idx_after>H-1]=H-1
            row_idx_after[row_idx_after<0]=0

            img_num, row_num, column_num = row_idx_after.shape

            row_idx_after = row_idx_after.reshape(row_idx_after.shape[0], -1)  # [34, 128]

            # assert False
            spline_poses = self.get_pose(i, img_idx, args)  # [34, 400, 3, 4] 
            
            if args.linear == False:
                spline_poses = spline_poses.reshape(img_num, H, 3, 4)

            spline_poses = spline_poses[torch.arange(args.pic_num).unsqueeze(1), row_idx_after, :, :] # [34, 128, 3, 4]
            
            spline_poses = spline_poses.reshape(img_num, row_num, column_num, 3, 4) # [34, 64, 2, 3, 4]

        else:
            # print(i, img_idx)
            spline_poses = self.get_pose(i, img_idx, args)  # [34, 400, 3, 4] 

            if args.linear == False:
                spline_poses = spline_poses.reshape(args.pic_num, H, 3, 4)

            spline_poses = spline_poses[:, row_idx]  # [34, 64, 3, 4]

        ret, point_index = self.render(i, img_idx, multisample_count, spline_poses, row_idx, point_idx, H, W, K, args, near=0, far=1.0, ray_idx_tv=None, training=True)

        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0 or i == self.special_iter:
        # if (i % args.i_img == 0 or i % args.i_novel_view == 0):
            if args.deblur_images % 2 == 0:
                all_poses = self.get_pose_even(i, torch.arange(self.se3.weight.shape[0]), args.deblur_images)
            else:
                all_poses = self.get_pose(i, torch.arange(self.se3.weight.shape[0]), args)
            return ret, point_index, spline_poses, all_poses
        else:
            return ret, point_index, spline_poses

    def get_pose(self, i, img_idx, args):

        return i

    def get_gt_pose(self, poses, args):

        return poses

    def render(self, barf_i, img_idx, multisample_count, sampled_poses, row_idx, point_idx, H, W, K, args, near=0., far=1., ray_idx_tv=None, training=False):


        N_rowpoint = point_idx.shape[1]
        N_row = row_idx.shape[0]
        if training:

            j = row_idx.unsqueeze(0).unsqueeze(-1).repeat(args.pic_num, 1, N_rowpoint).reshape(-1)  # [34*64*2]
            i = point_idx.unsqueeze(0).repeat(args.pic_num, 1, 1).reshape(-1)  # [34*64*2]
            row_idx_ = row_idx.unsqueeze(1).repeat(1, N_rowpoint).reshape(-1)
            point_idx_ = point_idx.reshape(-1)
            points = row_idx_ * W + point_idx_

            if multisample_count is None:
                sampled_poses = sampled_poses.unsqueeze(2).repeat(1, 1, N_rowpoint, 1, 1)  # [34*64*2, 3, 4]
            
            sampled_poses = sampled_poses.reshape(-1, 3, 4)

            rays_o_, rays_d_ = get_specific_rays(i, j, K, sampled_poses)  # [4352, 3]

            rays_o_d = torch.stack([rays_o_, rays_d_], 0)
            batch_rays = torch.permute(rays_o_d, [1, 0, 2])
        
        else:
            assert False
            rays_list = []
            for p in poses[:, :3, :4]:
                rays_o_, rays_d_ = get_rays(H, W, K, p)
                rays_o_d = torch.stack([rays_o_, rays_d_], 0)
                rays_list.append(rays_o_d)

            rays = torch.stack(rays_list, 0)
            rays = rays.reshape(-1, 2, H * W, 3)
            rays = torch.permute(rays, [0, 2, 1, 3])

            batch_rays = rays[:, ray_idx]
        
        batch_rays = batch_rays.reshape(-1, 2, 3)  # [4872, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, 4872, 3]

        # get standard rays
        rays_o, rays_d = batch_rays  # [4872, 3], [4872, 3]
        if args.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # [4872, 3]

        # sh = rays_d.shape
        if args.ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)  # [4872, 3], [4872, 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()  # [4872, 3]
        rays_d = torch.reshape(rays_d, [-1, 3]).float()  # [4872, 3]

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [4872, 1], [4872, 1]
        rays = torch.cat([rays_o, rays_d, near, far], -1)  # [4872, 8(3+3+1+1)]

        if args.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)  # [4872, 11]

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, args.N_samples])

        # perturb
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        raw_output = self.nerf.forward(barf_i, pts, viewdirs, args)

        rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf.raw2output(raw_output, z_vals, rays_d)

        if args.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw_output = self.nerf_fine.forward(barf_i, pts, viewdirs, args)
            rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf_fine.raw2output(raw_output, z_vals, rays_d)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['sigma'] = sigma

        return ret, points


    def render_4rendervideo(self, barf_i, poses, ray_idx, H, W, K, args, near=0., far=1., ray_idx_tv=None, training=False):

        '''
            barf_i: [1]
            poses: [203, 3, 4]
            ray_idx: [24]
        '''
        
        if training:
            assert False
            # print(ray_idx[:50])
            ray_idx_ = ray_idx.repeat(poses.shape[0])  # [4872 (24*203)] 
            # 每24个是一组，重复203遍，24个里面是每个位置的2d ray index，就是第几个2d点
            poses = poses.unsqueeze(1).repeat(1, ray_idx.shape[0], 1, 1).reshape(-1, 3, 4)  # [4872, 3, 4]
            # 每203个是一组，重复24遍，203个里面是每个相机位置的相机外参
            j = ray_idx_.reshape(-1, 1).squeeze() // W  # [4872 (24*203)]
            # 每24个是一组，重复203遍，24个里面是每个位置的j坐标
            i = ray_idx_.reshape(-1, 1).squeeze() % W  # [4872 (24*203)]
            # 每24个是一组，重复203遍，24个里面是每个位置的i坐标

            # print(i.shape, j.shape, K.shape, poses.shape)
            # assert False
            rays_o_, rays_d_ = get_specific_rays(i, j, K, poses)  # [4872, 3], [4872, 3]
            rays_o_d = torch.stack([rays_o_, rays_d_], 0)  # [2, 4872, 3]
            batch_rays = torch.permute(rays_o_d, [1, 0, 2])  # [4872, 2, 3]
        
        else:
            rays_list = []
            for p in poses[:, :3, :4]:
                rays_o_, rays_d_ = get_rays(H, W, K, p)
                rays_o_d = torch.stack([rays_o_, rays_d_], 0)
                rays_list.append(rays_o_d)

            rays = torch.stack(rays_list, 0)
            rays = rays.reshape(-1, 2, H * W, 3)
            rays = torch.permute(rays, [0, 2, 1, 3])

            batch_rays = rays[:, ray_idx]
        
        batch_rays = batch_rays.reshape(-1, 2, 3)  # [4872, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, 4872, 3]

        # get standard rays
        rays_o, rays_d = batch_rays  # [4872, 3], [4872, 3]
        if args.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # [4872, 3]

        # sh = rays_d.shape
        if args.ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)  # [4872, 3], [4872, 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()  # [4872, 3]
        rays_d = torch.reshape(rays_d, [-1, 3]).float()  # [4872, 3]

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [4872, 1], [4872, 1]
        rays = torch.cat([rays_o, rays_d, near, far], -1)  # [4872, 8(3+3+1+1)]

        if args.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)  # [4872, 11]

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, args.N_samples])

        # perturb
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        '''
            barf_i: [1]
            pts: [4872, 64, 3]
            viewdirs: [4872, 3]
        '''

        raw_output = self.nerf.forward(barf_i, pts, viewdirs, args)

        '''
            raw_output: [4872, 64, 4]
        '''

        # assert False

        rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf.raw2output(raw_output, z_vals, rays_d)

        if args.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw_output = self.nerf_fine.forward(barf_i, pts, viewdirs, args)
            rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf_fine.raw2output(raw_output, z_vals, rays_d)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['sigma'] = sigma

        return ret


    @torch.no_grad()
    def render_video(self, barf_i, poses, H, W, K, args):
        all_ret = {}
        ray_idx = torch.arange(0, H*W)
        for i in range(0, ray_idx.shape[0], args.chunk):
            ret = self.render_4rendervideo(barf_i, poses, ray_idx[i:i+args.chunk], H, W, K, args)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list([H, W]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        return all_ret

    @torch.no_grad()
    def forward_distancemap(self, i, multisample_count, img_idx, row_idx, point_idx, poses_num, H, W, K, args, novel_view=False):

        spline_poses = self.get_pose(i, img_idx, args)  # [34, 400, 3, 4]  每个时刻的两端的位姿
        spline_poses = spline_poses[:, row_idx]  # [34, 64, 3, 4]

        '''
            i: [1]
            spline_poses: [203 (29*7), 3, 4]
            ray_idx: [24 (args.N_rand=5000 // 203)]
        '''

        ret, point_index, xyz3d_pts = self.render_distancemap(i, img_idx, multisample_count, spline_poses, row_idx, point_idx, H, W, K, args, near=0, far=1.0, ray_idx_tv=None, training=True)

        return ret, point_index, spline_poses, xyz3d_pts

    @torch.no_grad()
    def render_distancemap(self, barf_i, img_idx, multisample_count, sampled_poses, row_idx, point_idx, H, W, K, args, near=0., far=1., ray_idx_tv=None, training=False):

        '''
            barf_i: [1]
            poses: [34, 400, 3, 4]
            row_idx: [64]
            point_idx: [64, args.N_rowpoint]
        '''

        N_rowpoint = point_idx.shape[1]

        j = row_idx.unsqueeze(0).unsqueeze(-1).repeat(args.pic_num, 1, N_rowpoint).reshape(-1)  # [34*64*2]
        i = point_idx.unsqueeze(0).repeat(args.pic_num, 1, 1).reshape(-1)  # [34*64*2]
        row_idx_ = row_idx.unsqueeze(1).repeat(1, N_rowpoint).reshape(-1)
        point_idx_ = point_idx.reshape(-1)
        points = row_idx_ * W + point_idx_
        sampled_poses = sampled_poses.unsqueeze(2).repeat(1, 1, N_rowpoint, 1, 1).reshape(-1, 3, 4)  # [34*64*2, 3, 4]
        rays_o_, rays_d_ = get_specific_rays(i, j, K, sampled_poses)  # [4352, 3]
        rays_o_d = torch.stack([rays_o_, rays_d_], 0)
        batch_rays = torch.permute(rays_o_d, [1, 0, 2])

        batch_rays = batch_rays.reshape(-1, 2, 3)  # [4872, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, 4872, 3]

        # get standard rays
        rays_o, rays_d = batch_rays  # [4872, 3], [4872, 3]
        if args.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # [4872, 3]

        # sh = rays_d.shape
        if args.ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)  # [4872, 3], [4872, 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()  # [4872, 3]
        rays_d = torch.reshape(rays_d, [-1, 3]).float()  # [4872, 3]

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [4872, 1], [4872, 1]
        rays = torch.cat([rays_o, rays_d, near, far], -1)  # [4872, 8(3+3+1+1)]

        if args.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)  # [4872, 11]

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, args.N_samples])

        # perturb
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        '''
            barf_i: [1]
            pts: [4872, 64, 3]
            viewdirs: [4872, 3]
        '''

        raw_output = self.nerf.forward(barf_i, pts, viewdirs, args)

        '''
            raw_output: [4872, 64, 4]
        '''

        # assert False

        rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf.raw2output(raw_output, z_vals, rays_d)

        if args.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            raw_output = self.nerf_fine.forward(barf_i, pts, viewdirs, args)
            rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf_fine.raw2output(raw_output, z_vals, rays_d)

            # print(rays_o.shape, rays_d.shape, depth_map.shape)
            # print(rays_o[..., None, :].shape, rays_d[..., None, :].shape, z_vals[..., :, None].shape)
            # assert False

            xyz3d_pts = rays_o + rays_d * depth_map[:, None]

            xyz3d_pts = self.ndcp2camp(xyz3d_pts, H, W, args.focal)

            # print(xyz3d_pts)

            xyz3d_rays_o = self.ndcp2camp(rays_o, H, W, args.focal)

            # print(xyz3d_rays_o)

            xyz3d_distances = torch.sqrt(((xyz3d_rays_o - xyz3d_pts) ** 2).sum(dim=1))

            # version2: 先过相机外参，然后求z的差值

            # print(distances)

            # assert False

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['sigma'] = sigma

        return ret, points, xyz3d_distances


    def ndcp2camp(self, points, W=None, H=None, f=None):

        z = 2/(points[:,2]-1)
        y = -(H*points[:,1]*z)/(2*f)
        x = -(W*points[:,0]*z)/(2*f)
        z = z.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        points_camera = torch.cat([x, y, z], dim=-1)

        return points_camera