import mayavi.mlab as mlab
from model.geometry import *
import os
import torch
import argparse
import cv2
from model.registration import Registration
import  yaml
from easydict import EasyDict as edict



def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')

    """demo data"""
    intrinsics = np.loadtxt(config.intrinsics)

    """load lepard predicted matches as landmarks"""
    data = np.load(config.correspondence)
    ldmk_src = data['src_pcd'][0][data['match'][:,1] ]
    ldmk_tgt = data['tgt_pcd'][0][data['match'][:,2] ]
    uv_src = xyz_2_uv(ldmk_src, intrinsics)
    uv_tgt = xyz_2_uv(ldmk_tgt, intrinsics)
    landmarks = ( torch.from_numpy(uv_src).to(config.device),
                  torch.from_numpy(uv_tgt).to(config.device))


    """init model with source frame"""
    model = Registration(config.src_depth, K=intrinsics, config=config)

    model.register_a_depth_frame( config.tgt_depth,  landmarks=landmarks)


