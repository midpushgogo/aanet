from __future__ import print_function
import argparse 
import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from  skimage import io
import numpy as np
import time
import sys
import random
#from models import *
import nets
from utils.preprocess import scale_disp, default_transform,scale_transform
from  torchvision import utils as vutils
from utils import utils
parser = argparse.ArgumentParser(description='predict single image')
parser.add_argument('--image', type=str, default='')

parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--loadmodel', type=str, default='')
parser.add_argument('--devices', type=str)
parser.add_argument('--savepath', type=str, default='')
parser.add_argument('--maxdisp', type=int, default=192)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--vis', action='store_true')


# AANet
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--combinecost', action="store_true", default=False)
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

model = nets.AANet(args.max_disp,
                   num_downsample=args.num_downsample,
                   feature_type=args.feature_type,
                   no_feature_mdconv=args.no_feature_mdconv,
                   feature_pyramid=args.feature_pyramid,
                   feature_pyramid_network=args.feature_pyramid_network,
                   feature_similarity='combind_volume' if args.combinecost else 'correlation',
                   aggregation_type=args.aggregation_type,
                   num_scales=args.num_scales,
                   num_fusions=args.num_fusions,
                   num_stage_blocks=args.num_stage_blocks,
                   num_deform_blocks=args.num_deform_blocks,
                   no_intermediate_supervision=args.no_intermediate_supervision,
                   refinement_type=args.refinement_type,
                   mdconv_dilation=args.mdconv_dilation,
                   deformable_groups=args.deformable_groups,
                   attention=args.attention)

model = nn.DataParallel(model, device_ids=devices)
model.cuda()

if args.loadmodel is not None:
    state=torch.load(args.loadmodel)
    new_state_dict={}
    weights=state['state_dict'] if 'state_dict' in state.keys() else state
    for k,v in weights.items():
        name='module.'+k if 'module' not in k else k
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict,strict=False)
    print('load pretrained model')

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)

def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        
    with torch.no_grad():
        time_start = time.time()
        output = model(imgL, imgR, False)
        output=output[-1]
        time_end = time.time()
        output = torch.squeeze(output, dim=1)
        output = torch.squeeze(output, dim=0)
        pred_disp = output.data.cpu().numpy()
    
    return pred_disp, time_end-time_start

def main():
    file=[]
    for i in os.listdir(args.image):
        file.append(i)
    print(file)
    t=0
    for i in file:

        imgL_ori = (io.imread(args.image+i+'/'+'im0.png'))
        imgR_ori = (io.imread(args.image+i+'/'+'im1.png'))
        rgb_transform = default_transform()

        imgL = rgb_transform(imgL_ori)
        imgR = rgb_transform(imgR_ori)
        print(imgL.shape)
        imgL = torch.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = torch.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
        kk=96
        if imgL.shape[2]%kk!=0:
            times = imgL.shape[2] // kk
            top_pad = (times + 1) * kk - imgL.shape[2]
        else:
            top_pad = 0
        if imgL.shape[3] % kk != 0:
            times = imgL.shape[3] // kk
            right_pad = (times + 1) * kk - imgL.shape[3]
        else:
            right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0))


        output, inf_time = test(imgL, imgR)
        output = output[top_pad:, :]
        if right_pad!=0:
            output=output[:,:-right_pad]
        if not os.path.exists(args.savepath+"/"+i):
                os.makedirs(args.savepath+'/'+i)
        write_pfm(args.savepath+'/'+i+ "/"+'disp0EDNet.pfm', output)
        with open(args.savepath+'/'+i+ "/"+'timeEDNet.txt',"w") as f:
            f.write(str(inf_time))
        if args.vis:
            vutils.save_image(torch.from_numpy(output), args.savepath+ '/'+i+'.jpg', normalize=True)

        print('finished prediction with inf time %.3f' % (inf_time))
        t+=inf_time
    print('time ',t/len(file))

if __name__=='__main__':
    main()







