
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms.functional as fn
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


DEVICE = 'cuda'
def same(cur_pts,cur_img,forw_img):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    args.model="/home/kangni/catkin_ws/src/VINS-MONO-20.04-/feature_tracker/src/RAFT/models/raft-things.pth"

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        image1=np.tile(cur_img,(3,1,1)).reshape(1,3,480,752)
        image2 = np.tile(forw_img, (3, 1, 1)).reshape(1, 3, 480, 752)
        image1=torch.from_numpy(image1).to(DEVICE)
        image2=torch.from_numpy(image2).to(DEVICE)
        image1=fn.resize(image1,size=(440,1024))
        image2=fn.resize(image2,size=(440,1024))



        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        flow_up=flow_up.to('cpu').numpy()
        cur_index=np.flip(cur_pts.reshape((-1,2))*np.array([1024/752,440/480]),axis=1).astype(int)
        grad_up = flow_up[0,:,cur_index[:, 0], cur_index[:, 1]]
        grad_up = np.flip(grad_up,axis=1)
        grad_up = grad_up*np.array([752/1024,480/440])*46.6

        flow_low = flow_low.to('cpu').numpy()
        cur_index = np.flip(cur_pts.reshape((-1, 2)) * np.array([1024 / 752, 440 / 480]), axis=1).astype(int)
        grad_low = flow_low[0, :, cur_index[:, 0], cur_index[:, 1]]
        grad_low = np.flip(grad_low, axis=1)
        grad_low = grad_low * np.array([752 / 1024, 480 / 440]) * 46.6
        #grad = np.flip(grad, axis=1)
        output=cur_pts +grad_low/2+grad_up/2
    return output







