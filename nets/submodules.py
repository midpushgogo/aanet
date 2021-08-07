# freda (todo) : 

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
# from correlation_package.modules.corr import Correlation1d # from PWC-Net
# from layers_package.channelnorm_package.channelnorm import ChannelNorm
# from layers_package.resample2d_package.resample2d import Resample2d
import copy


class DynamicConv2d(nn.Module):

    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, in_channel, out_channel=None):
        if out_channel:
            return self.conv.weight[:out_channel, :in_channel, :, :]
        else:
            return self.conv.weight[:, :in_channel, :, :]

    def forward(self, x):
        in_channel = x.size(1)
        filters = self.get_active_filter(in_channel).contiguous()

        def get_same_padding(kernel_size):
            if isinstance(kernel_size, tuple):
                assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
                p1 = get_same_padding(kernel_size[0])
                p2 = get_same_padding(kernel_size[1])
                return p1, p2
            assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
            assert kernel_size % 2 > 0, 'kernel size should be odd number'
            return kernel_size // 2

        padding = get_same_padding(self.kernel_size)
        # filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DyRes(nn.Module):
    def __init__(self, max_in=98, max_out=128, stride=1):
        super(DyRes, self).__init__()
        self.conv1 = DynamicConv2d(max_in, max_out, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(max_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(max_out, max_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(max_out)
        self.stride = stride
        self.max_out = max_out

        if stride != 1 or max_out != max_in:
            self.shortcut = nn.Sequential(
                DynamicConv2d(max_in, max_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(max_out))
        else:
            self.shortcut = None

    def fix_dynamic(self, in_channels=72):

        tmp_dy = copy.deepcopy(self.conv1)
        tmp_fix = nn.Conv2d(in_channels, self.max_out, kernel_size=3, padding=1)
        tmp_fix.weight.data = tmp_dy.get_active_filter(in_channels)
        tmp_fix.cuda()
        self.conv1 = copy.deepcopy(tmp_fix)
        if self.shortcut is not None:
            tmp_fix = nn.Conv2d(in_channels, self.max_out, kernel_size=1, stride=self.stride)
            tmp_dy = copy.deepcopy(self.shortcut[0])
            tmp_fix.weight.data = tmp_dy.get_active_filter(in_channels)
            tmp_fix.cuda()
            self.shortcut[0] = tmp_fix

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
        )


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


def predict_flow(in_planes, out_planes=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def build_corr(img_left, img_right, max_disp=40, zero_volume=None):
    B, C, H, W = img_left.shape
    if zero_volume is not None:
        tmp_zero_volume = zero_volume  # * 0.0
        # print('tmp_zero_volume: ', mean)
        volume = tmp_zero_volume
    else:
        volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if (i > 0) & (i < W):
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :W - i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume


def build_comb(left_feature, right_feature, max_disp=40, zero_volume=None, comb=None):
    b, c, h, w = left_feature.shape
    cost_volume_correlation = left_feature.new_zeros(b, max_disp, h, w)
    cost_volume_con = left_feature.new_zeros(b, 2 * c, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume_correlation[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                    right_feature[:, :, :, :-i]).mean(dim=1)
            cost_volume_con[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                        dim=1)
        else:
            cost_volume_correlation[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
            cost_volume_con[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

    # cost volume squeezement
    cost_volume_concat = comb(cost_volume_con).squeeze(dim=1)  # N C D H W -> N D H W

    cost_volume = cost_volume_concat + cost_volume_correlation  # N D H W

    cost_volume = cost_volume.contiguous()
    return cost_volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0))
        shifted_right = F.pad(
            torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
                             requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def channel_normalize(x):
    return x / (torch.norm(x, 2, dim=1, keepdim=True) + 1e-8)


def channel_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + 1e-8)


def warp_right_to_left(x, disp, warp_grid=None):
    # print('size: ', x.size())

    B, C, H, W = x.size()
    # mesh grid
    if warp_grid is not None:
        xx0, yy = warp_grid
        xx = xx0 + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
    else:
        # xx = torch.arange(0, W, device=disp.device).float()
        # yy = torch.arange(0, H, device=disp.device).float()
        xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
        yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
        # if x.is_cuda:
        #    xx = xx.cuda()
        #    yy = yy.cuda()
        xx = xx.view(1, -1).repeat(H, 1)
        yy = yy.view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        # apply disparity to x-axis
        xx = xx + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
        yy = 2.0 * yy / max(H - 1, 1) - 1.0

    grid = torch.cat((xx, yy), 1)

    vgrid = grid
    # vgrid[:, 0, :, :] = vgrid[:, 0, :, :] + disp[:, 0, :, :]
    # vgrid[:, 0, :, :].add_(disp[:, 0, :, :])
    # vgrid.add_(disp)

    # scale grid to [-1,1] 
    # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1)-1.0
    # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1)-1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    # mask = torch.autograd.Variable(torch.ones_like(x))
    # mask = nn.functional.grid_sample(mask, vgrid)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1

    # return output*mask
    return output  # *mask


if __name__ == '__main__':
    x = torch.rand([2, 3, 540, 960])
    disp = torch.ones([2, 1, 540, 960]) * 5
    dummy_y = torch.zeros([2, 1, 540, 960])
    x = x.cuda()
    disp = disp.cuda()
    dummy_y = dummy_y.cuda()

    # test channel normalization
    # cn_fn = channel_length(x)
    # cn_layer = ChannelNorm()
    # output_cn = cn_layer(x)
    # print(cn_fn[:, :, :10, :10])
    # print(output_cn[:, :, :10, :10])
    # print(cn_fn.size())
    # print(output_cn.size())
    # print(torch.norm(cn_fn - output_cn, 2, 1).mean())

    warpped_left = warp_right_to_left(x, -disp)
    # warp_layer = Resample2d()
    # output_warpped = warp_layer(x, -torch.cat((disp, dummy_y), dim = 1))
    # print(torch.norm(warpped_left - output_warpped, 2, 1).mean())
    save_image(x[0], 'img0.png')
    save_image(warpped_left[0], 'img1.png')
    # save_image(output_warpped[0], 'img2.png')
