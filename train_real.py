import torch

from nerf import *
import optimize_pose_linear
import matplotlib.pyplot as mp
import random
import cv2
from mpl_toolkits.mplot3d import Axes3D
from load_blender import load_llff_data_real

import tensorboardX

import Spline
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


def get_pose_pixel_ratio_forward(flows):
    flow_num, flow_h, flow_w, _ = flows.shape
    row_grid = torch.linspace(0, flow_h - 1, flow_h).unsqueeze(-1).repeat(1, flow_w)
    row_disp = flows[:, :, :, 1]
    row_in_first = row_grid
    row_in_second = row_grid + row_disp
    pose_disp = row_in_second + (flow_h - row_in_first)
    max_pixel_disp, _ = torch.max(torch.abs(flows), dim=-1)
    pose_pixel_ratio = (pose_disp / max_pixel_disp).int()

    return pose_pixel_ratio


def get_pose_pixel_ratio_backward(flows):
    flow_num, flow_h, flow_w, _ = flows.shape
    row_grid = torch.linspace(0, flow_h - 1, flow_h).unsqueeze(-1).repeat(1, flow_w)
    row_disp = flows[:, :, :, 1]
    row_in_first = row_grid
    row_in_second = row_grid + row_disp
    pose_disp = (flow_h - row_in_second) + row_in_first
    max_pixel_disp, _ = torch.max(torch.abs(flows), dim=-1)
    pose_pixel_ratio = (pose_disp / max_pixel_disp).int()

    return pose_pixel_ratio


def se3_2_t_parallel(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    # A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    # R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q_parallel(w)
    return q, t.squeeze(-1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():

    setup_seed(20230823)
    
    parser = config_parser()
    args = parser.parse_args()
    print('spline numbers: ', args.deblur_images)
    if args.multisample > 0:
        print('Using multi-sampling!')

    # Load data images and groundtruth
    K = None
    gs_images = None
    if args.dataset_type == 'llff':
        images_all, poses, bds, render_poses, i_test = load_llff_data_real(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images_all.shape, render_poses.shape, hwf, args.datadir)

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images_all.shape[0])[::args.llffhold][1:]
        
        i_train = np.array([i for i in np.arange(int(images_all.shape[0]))])

        i_test = np.array([i for i in np.arange(int(images_all.shape[0]))])
        
        i_val = i_test

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    images = torch.tensor(images_all[i_train])

    poses = torch.tensor(poses)

    poses_train = poses[i_train][:, :3, :4]
    poses_start_se3 = SE3_to_se3_N(poses_train)  # [34, 6]

    poses_end_se3 = poses_start_se3

    poses_org = poses.repeat(args.deblur_images, 1, 1)  # [29*7, 3, 5]
    # print(poses_org.shape)
    poses = poses_org[:, :, :4]  # [29*7, 3, 4]
    poses_num = poses.shape[0]
    # print(poses.shape)
    # assert False

    print('Loaded', images.shape, render_poses.shape, hwf, args.datadir)

    # assert False

    print('DEFINING BOUNDS')

    if args.no_ndc:
        assert False
        near = torch.min(bds_start) * .9
        far = torch.max(bds_start) * 1.
    else:
        near = 0.
        far = 1.
    
    print('NEAR FAR', near, far)
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    args.focal = focal
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    print_file = os.path.join(basedir, expname, 'print.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.load_weights:

        print('Linear Spline Model Loading!')
        model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)  # [29, 6]
        
        graph = model.build_network(args)
        optimizer, optimizer_se3 = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)
        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
        global_step = graph_ckpt['global_step']

    else:

        low, high = 0.0001, 0.005
        rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
        poses_start_se3 = poses_start_se3 + rand

        model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)  # [29, 6]

        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3 = model.setup_optimizer(args)
        init_nerf(graph.nerf)
        init_nerf(graph.nerf_fine)

    N_iters = args.N_iters + 1

    args.pic_num = images.shape[0]

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0

    if not args.load_weights:
        global_step = start

    global_step_ = global_step
    threshold = N_iters + 1

    train_writer = tensorboardX.SummaryWriter(os.path.join(basedir, expname, 'tensorboard'))

    i_row_batch = 0
    i_columns_batch = 0

    all_rows = torch.randperm(H)
    all_columns = torch.randperm(W)

    forward_flows = []
    backward_flows = []

    raft = torch.nn.DataParallel(RAFT())
    raft.load_state_dict(torch.load('raft_models/raft-things.pth'))

    raft = raft.module
    raft.to('cuda')
    raft.eval()

    with torch.no_grad():
        for i in tqdm(range(images.shape[0] - 1)):
            image1 = (images[i:(i+1)].permute(0, 3, 1, 2).cpu().numpy() * 255).astype(np.uint8)
            image2 = (images[(i+1):(i+2)].permute(0, 3, 1, 2).cpu().numpy() * 255).astype(np.uint8)

            image1 = torch.from_numpy(image1).float().to('cuda')
            image2 = torch.from_numpy(image2).float().to('cuda')

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_forward = raft(image1, image2, iters=20, test_mode=True)
            _, flow_backward = raft(image2, image1, iters=20, test_mode=True)

            forward_flows.append(flow_forward[0])
            backward_flows.append(flow_backward[0])
        
    forward_flows = torch.stack(forward_flows).permute(0, 2, 3, 1)
    backward_flows = torch.stack(backward_flows).permute(0, 2, 3, 1)
    
    forward_ratios = get_pose_pixel_ratio_forward(forward_flows)
    backward_ratios = get_pose_pixel_ratio_backward(backward_flows)

    forward_ratios = torch.cat([forward_ratios, torch.zeros_like(forward_ratios[0:1])], dim=0)
    backward_ratios = torch.cat([torch.zeros_like(backward_ratios[0:1]), backward_ratios], dim=0)

    multisample_count = torch.stack([backward_ratios, forward_ratios], dim=-1)


    ###########################
    estimated_poses = graph.se3.weight.data
    se3_start, se3_end = estimated_poses[:, :6].unsqueeze(1), estimated_poses[:, 6:].unsqueeze(1)
    pose_nums = torch.tensor([0, args.deblur_images]).reshape(1, -1).repeat(1, 1)
    spline_poses_viz = Spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
    spline_poses_viz = spline_poses_viz.reshape(-1, 3, 4)[:, :, 3].detach().cpu().numpy()
    estimated_pose = np.concatenate([spline_poses_viz[::2], spline_poses_viz[-1:]], axis=0)
    fig = mp.figure('3D Scatter')
    ax3d = fig.add_axes(Axes3D(fig))
    ax3d.set_xlabel('X', fontsize=14)
    ax3d.set_ylabel('Y', fontsize=14)
    ax3d.set_zlabel('Z', fontsize=14)
    ax3d.scatter(spline_poses_viz[: ,0], spline_poses_viz[: ,1], spline_poses_viz[: ,2], s=5, alpha=1, c='r')
    for viz_i in range(spline_poses_viz.shape[0] // 2):
        ax3d.plot(spline_poses_viz[viz_i*2:viz_i*2+2, 0], spline_poses_viz[viz_i*2:viz_i*2+2, 1], spline_poses_viz[viz_i*2:viz_i*2+2, 2], color='r', linewidth=1)
    os.makedirs(os.path.join(basedir, expname, 'viz_poses'), exist_ok=True)
    mp.savefig(os.path.join(basedir, expname, 'viz_poses', f'00000_init.png'))
    fig.clf()
    #################################

    for i in trange(start, threshold):

    ### core optimization loop ###
        i = i+global_step_

        img_idx = torch.randperm(images.shape[0])


        row_batch = all_rows[i_row_batch:i_row_batch+args.N_row]
        i_row_batch += args.N_row
        if i_row_batch + args.N_row >= H:
            rand_idx = torch.randperm(all_rows.shape[0])
            all_rows = all_rows[rand_idx]
            i_row_batch = 0
        
        assert args.N_rowpoint * args.N_row < W

        column_batch = all_columns[i_columns_batch:i_columns_batch+(args.N_row * args.N_rowpoint)]
        column_batch = column_batch.reshape(args.N_row, args.N_rowpoint)
        i_columns_batch += args.N_row * args.N_rowpoint
        if i_columns_batch + args.N_row * args.N_rowpoint > W:
            rand_idx = torch.randperm(all_columns.shape[0])
            all_columns = all_columns[rand_idx]
            i_columns_batch = 0

        if i < 50000:
            mc = multisample_count
        else:
            mc = None
        
        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0 or i == 10000:
        # if (i % args.i_img == 0 or i % args.i_novel_view == 0):
            ret, point_index, spline_poses, all_poses = graph.forward(i, mc, img_idx, row_batch, column_batch, poses_num, H, W, K, args)
        else:
            ret, point_index, spline_poses = graph.forward(i, mc, img_idx, row_batch, column_batch, poses_num, H, W, K, args)

        img_num, h_img, w_img, c_img = images.shape
        target_s = images[img_idx].reshape(-1, h_img * w_img, 3)  # [29, 400*400, 3]
        target_s = target_s[:, point_index]  # [29, 24, 3]
        target_s = target_s.reshape(-1, 3)  # [696(29*24), 3]

        rgb_blur = ret['rgb_map']

        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        

        img_loss = img2mse(rgb_blur, target_s)

        _, start_ts = se3_2_t_parallel(graph.se3.weight[:, :6])

        _, end_ts = se3_2_t_parallel(graph.se3.weight[:, 6:])
        
        all_points = torch.stack([start_ts, end_ts], dim=1).reshape(-1, 3)

        one_direc = all_points[:-2][1::2] - all_points[:-2][::2]
        two_direc = all_points[2:][1::2] - all_points[:-2][::2]
        mid_direc = all_points[1:-1][1::2] - all_points[1:-1][::2]

        one_direc = one_direc / torch.norm(one_direc, dim=1, keepdim=True)
        mid_direc = mid_direc / torch.norm(mid_direc, dim=1, keepdim=True)
        two_direc = two_direc / torch.norm(two_direc, dim=1, keepdim=True)

        smooth_loss = F.mse_loss(mid_direc, (one_direc + two_direc) / 2.)

        pose_loss = torch.tensor(0)

        loss = img_loss + pose_loss + smooth_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            extras_blur = ret['rgb0']
            img_loss0 = img2mse(extras_blur, target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()
        optimizer_se3.step()

        train_writer.add_scalar('loss', loss.item(), i)
        train_writer.add_scalar('img_loss', img_loss.item(), i)
        train_writer.add_scalar('pose_loss', pose_loss.item(), i)
        train_writer.add_scalar('smooth_loss', smooth_loss.item(), i)
        train_writer.add_scalar('PSNR', psnr.item(), i)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose
        ###############################

        if i % args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} img_loss: {img_loss.item()} pose_loss: {pose_loss.item()}  smooth_loss: {smooth_loss.item()} coarse_loss: {img_loss0.item()}, PSNR: {psnr.item()}")
            with open(print_file, 'a') as outfile:
                outfile.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} img_loss: {img_loss.item()} pose_loss: {pose_loss.item()} smooth_loss: {smooth_loss.item()} coarse_loss: {img_loss0.item()}, PSNR: {psnr.item()}\n")

        if i < 10:
            print('coarse_loss:', img_loss0.item())
            print('pose_loss:', pose_loss.item())
            print('smooth_loss:', smooth_loss.item())
            with open(print_file, 'a') as outfile:
                outfile.write(f"coarse loss: {img_loss0.item()}\n")
                outfile.write(f"pose loss: {pose_loss.item()}\n")
                outfile.write(f"smooth_loss: {smooth_loss.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_vizpose == 0 or i <= 10001 and i % 1000 == 0:
            estimated_poses = graph.se3.weight.data
            se3_start, se3_end = estimated_poses[:, :6].unsqueeze(1), estimated_poses[:, 6:].unsqueeze(1)
            pose_nums = torch.tensor([0, args.deblur_images]).reshape(1, -1).repeat(1, 1)
            
            spline_poses_viz = Spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
            spline_poses_viz = spline_poses_viz.reshape(-1, 3, 4)[:, :, 3].detach().cpu().numpy()

            estimated_pose = np.concatenate([spline_poses_viz[::2], spline_poses_viz[-1:]], axis=0)

            fig = mp.figure('3D Scatter')
            ax3d = fig.add_axes(Axes3D(fig))
            ax3d.set_xlabel('X', fontsize=14)
            ax3d.set_ylabel('Y', fontsize=14)
            ax3d.set_zlabel('Z', fontsize=14)

            ax3d.scatter(spline_poses_viz[: ,0], spline_poses_viz[: ,1], spline_poses_viz[: ,2], s=5, alpha=1, c='r')

            for viz_i in range(spline_poses_viz.shape[0] // 2):
                ax3d.plot(spline_poses_viz[viz_i*2:viz_i*2+2, 0], spline_poses_viz[viz_i*2:viz_i*2+2, 1], spline_poses_viz[viz_i*2:viz_i*2+2, 2], color='r', linewidth=1)

            os.makedirs(os.path.join(basedir, expname, 'viz_poses'), exist_ok=True)
            mp.savefig(os.path.join(basedir, expname, 'viz_poses', f'{str(i).zfill(5)}.png'))

            fig.clf()


        if i % args.i_img == 0 and i > 0 or i == 10000:
            
            img_dir = os.path.join(args.basedir, args.expname, 'img_test_out', str(i).zfill(5))
            with torch.no_grad():
                imgs_render = render_image_test(i, graph, poses[i_test], H, W, K, args, img_dir)

        if i % args.i_video == 0 and i > 0 or i == 10000:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
