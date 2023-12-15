import os
from os.path import join as pjoin

import torch

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE
from data.t2m_dataset import Text2MotionDataset

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed
from torch.utils.data import DataLoader
from visualization.joints2bvh import Joint2BVHConvertor

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    eval_dir = pjoin('eval_results', opt.dataset_name, opt.name, opt.ext)
    joints_dir = pjoin(eval_dir, 'joints')
    animation_dir = pjoin(eval_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    if opt.use_res_model:
        res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
        res_opt = get_opt(res_opt_path, device=opt.device)
        res_model = load_res_model(res_opt)

        res_model.eval()
        res_model.to(opt.device)
        assert res_opt.vq_name == model_opt.vq_name

    opt.data_root = './dataset/HumanML3D'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    test_split_file = pjoin(opt.data_root, 'test.txt')


    test_dataset = Text2MotionDataset(opt, mean, std, test_split_file)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # print('loading checkpoint {}'.format(file))
    t2m_transformer = load_trans_model(model_opt, 'latest.tar')
    t2m_transformer.eval()
    vq_model.eval()

    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    sample = 0

    kinematic_chain = t2m_kinematic_chain

    converter = Joint2BVHConvertor()

    for i, batch in enumerate(test_loader):
        print("Batch %d"%i)
        if i >= opt.num_batch:
            break
        captions, motion, m_length = batch
        motion = motion.to(opt.device).float()
        m_length = m_length.to(opt.device).long()

        for r in range(opt.repeat_times):
            print("-->Repeat %d"%r)
            with torch.no_grad():
                mids = t2m_transformer.generate(captions, m_length//4,
                                                timesteps=opt.time_steps,
                                                cond_scale=opt.cond_scale,
                                                temperature=opt.temperature,
                                                topk_filter_thres=opt.topkr,
                                                gsample=opt.gumbel_sample,
                                                force_mask=opt.force_mask)
                # print(mids)
                # print(mids.shape)
                if opt.use_res_model:
                    mids = res_model.generate(mids, captions, m_length // 4, temperature=1, cond_scale=5)
                else:
                    mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                pred_motions = pred_motions.detach().cpu().numpy()

                data = test_dataset.inv_transform(pred_motions)

                if r==0:
                    vq_motion, _, _ = vq_model(motion)
                    gt_data = test_dataset.inv_transform(vq_motion.cpu().numpy())

            for k, (caption, joint_data)  in enumerate(zip(captions, data)):
                print("---->Sample %d"%k)

                if r==0:
                    gt_joint_data = gt_data[k, :m_length[k]]
                    gt_joint = recover_from_ric(torch.from_numpy(gt_joint_data).float(), 22).numpy()
                    save_name = "batch%d_sample%d"%(i, k)
                    plot_3d_motion(pjoin(animation_dir, "%s.mp4"%save_name), kinematic_chain, gt_joint, title=caption, fps=20)
                    np.save(pjoin(joints_dir, "%s.npy"%save_name), gt_joint)

                    bvh_path = pjoin(animation_dir, "batch%d_sample%d.bvh" % (i, k))
                    _, ik_joint = converter.convert_sgd(gt_joint, filename=bvh_path, iterations=50)

                joint_data = joint_data[:m_length[k]]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

                bvh_path = pjoin(animation_dir, "batch%d_sample%d_repeat%d_len%d_ik.bvh"%(i, k, r, m_length[k]))
                _, ik_joint = converter.convert_sgd(joint, filename=bvh_path, iterations=50)

                bvh_path = pjoin(animation_dir, "batch%d_sample%d_repeat%d_len%d.bvh" % (i, k, r, m_length[k]))
                _, _ = converter.convert_sgd(joint, filename=bvh_path, iterations=50, foot_ik=False)

                ik_save_path = pjoin(animation_dir, "batch%d_sample%d_repeat%d_len%d_ik.mp4"%(i, k, r, m_length[k]))
                save_path = pjoin(animation_dir, "batch%d_sample%d_repeat%d_len%d.mp4"%(i, k, r, m_length[k]))


                plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
                np.save(pjoin(joints_dir, "batch%d_sample%d_repeat%d_len%d.npy"%(i, k, r, m_length[k])), joint)
                np.save(pjoin(joints_dir, "batch%d_sample%d_repeat%d_len%d_ik.npy"%(i, k, r, m_length[k])), ik_joint)