import torch
import torch.nn as nn
import torch.nn.functional as F

from deform.modules.modulated_deform_conv import ModulatedDeformConvPack

from nets.feature import BasicBlock, BasicConv, Conv2x
from nets.deform import DeformConv2d
from nets.warp import disp_warp
from torchvision import utils as vutils
class SA_Module(nn.Module):
    """
    Note: simple but effective spatial attention module.
    """

    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_value = self.attention_value(x)

        return attention_value


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class StereoNetRefinement(nn.Module):
    def __init__(self,attention):
        super(StereoNetRefinement, self).__init__()

        # Original StereoNet: left, disp
        if attention:
            
            self.conv = conv2d(10, 32)
        else:
            self.conv=conv2d(4,32)
        self.attention=attention
        self.attentionnet = SA_Module(input_nc=10)
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img=None):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, H, W]
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
        """
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp * scale_factor  # scale correspondingly

        if self.attention:
            warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
            error = warped_right - left_img  # [B, C, H, W]
            query = torch.cat((left_img, right_img, error, disp), dim=1)
            attention_map=self.attentionnet(query)
          #  concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
            out = self.conv(query*attention_map)
        else:
            concat = torch.cat((disp, left_img), dim=1)  # [B, 4, H, W]
            out = self.conv(concat)
        out = self.dilated_blocks(out)
        residual_disp = self.final_conv(out)

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp

class Occlusion_residual(nn.Module):
    def __init__(self, in_channel, feature_channel=32, deform=False):
        super(Occlusion_residual, self).__init__()
        if deform:
            self.occlusion_residual = nn.Sequential(
                nn.Conv2d(in_channel, feature_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature_channel),
                nn.ReLU(True),
                ModulatedDeformConvPack(feature_channel, feature_channel*2, kernel_size=(3, 3), stride=1,
                                        padding=1, deformable_groups=2),
                nn.BatchNorm2d(feature_channel*2),
                nn.ReLU(True),
                nn.Conv2d(feature_channel*2, feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature_channel*2),
                nn.ReLU(True),
                nn.Conv2d(feature_channel*2, 1, kernel_size=3, stride=1, padding=1, bias=False)
            )
        else:
            self.occlusion_residual = nn.Sequential(

                    nn.Conv2d(in_channel+1, feature_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channel),
                    nn.ReLU(True),
                    nn.Conv2d(feature_channel, feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channel*2),
                    nn.ReLU(True),
                    nn.Conv2d(feature_channel*2, feature_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channel*2),
                    nn.ReLU(True),
                    nn.Conv2d(feature_channel*2, 1, kernel_size=3, stride=1, padding=1, bias=False)
                )

    def forward(self, input_feature):
        occlusion_residual = self.occlusion_residual(input_feature)
        return occlusion_residual

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
    )

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
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



class Aggregation2D(nn.Module):
    def __init__(self, disp_range, feature_channel=32, in_feature=None):
        super().__init__()
        if in_feature is not None:
            C = in_feature
        else:
            C = 0
        self.aggregation = convbn(disp_range + C, feature_channel, kernel_size=3, stride=1, pad=1, dilation=1)
        self.conv1 = ResBlock(feature_channel, feature_channel*2, stride=1)
        self.conv2 = ResBlock(feature_channel*2, feature_channel*2, stride=1)
        self.pred = nn.Conv2d(feature_channel*2, 1, 3, 1, 1, bias=False)

    def forward(self, cost_volume, in_feature=None):
        if in_feature is not None:
            cost_volume = torch.cat((cost_volume, in_feature), dim=1)
        output = self.aggregation(cost_volume)
        output = self.conv1(output)  # 1/2
        output = self.conv2(output)            # 1/2
        output = self.pred(output)
        return output

def res_dynamic_cost_volume(offset, left_feature, right_feature, disp, mode='correlation'):
    '''Args:
            offset: number of default search range which is supposed to be an even number
            left_feature: [N, C, H, W]
            right_feature: [N, C, H, W]
            disp: [N,1,H,W]
            mode: correlation or concatenation
        return:
            cost_volume: [N, num_offset, H, W]
    '''

    N, C, H, W = left_feature.shape
    if mode == 'correlation':
        res_volume = torch.zeros(N, offset, H, W).to(left_feature.device)
    origin_disp = disp.clone().repeat(1, offset, 1, 1)
    # provide origin search range
    if offset != 0:
        if offset % 2 == 0:
            search_range = torch.arange(-offset//2, offset//2, dtype=torch.float, device=left_feature.device).view(1, offset, 1, 1)
        else:
            search_range = torch.arange(-(offset-1)//2, (offset+1)//2, dtype=torch.float, device=left_feature.device).view(1, offset, 1, 1)
    else:
        search_range = 0

    grid_x = torch.arange(H, dtype=torch.float)
    grid_y = torch.arange(W, dtype=torch.float)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y)

    grid_coor = torch.stack((grid_y, grid_x), dim=-1).view(1, H, W, 2).unsqueeze(dim=1).repeat(N, offset, 1, 1, 1).to(left_feature.device)
    grid_coor[:, :, :, :, 0] += search_range - origin_disp

    grid_coor[:, :, :, :, 0] = 2 * (grid_coor[:, :, :, :, 0] / (W - 1)) - 1
    grid_coor[:, :, :, :, 1] = 2 * (grid_coor[:, :, :, :, 1] / (H - 1)) - 1
    for i in range(offset):
        right_feature_sampled = F.grid_sample(right_feature, grid_coor[:, i], mode='bilinear',align_corners=True, padding_mode='zeros')
        if mode == 'correlation':
            res_volume[:, i] = (left_feature * right_feature_sampled).mean(dim=1)

    return res_volume

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, input_feature):
        out = self.conv1_1(input_feature)
        out = self.conv3(out)
        out = self.conv1_2(out)
        return out


class Occlution_pred(nn.Module):
    def __init__(self, in_channel=7, feature_channel=32):
        """Feature extractor of StereoNet
        Args:
            input: concat[error_map, disparity, right_image]
        """
        super(Occlution_pred, self).__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(in_channel, feature_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(True),
            nn.Conv2d(feature_channel, feature_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_channel)
        )
        in_channel = feature_channel
        # for upsample
        dilation_list = [2, 4]
        dilated_blocks = nn.ModuleList()
        for dilation in dilation_list:
            dilated_blocks.append(ResidualBlock(in_channel, feature_channel, dilation=dilation))

        self.dilation_blocks = nn.Sequential(*dilated_blocks)
        self.pred = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.projector(img)  # [B, 32, H/8, W/8]
        identity = out

        for m in self.dilation_blocks:
            out = m(out)
            out = out + identity
            identity = out

        occlusion = self.pred(out)
        return occlusion


class RescostRefinement(nn.Module):
    def __init__(self,disp_range=9,mode="correlation"):
        super(RescostRefinement, self).__init__()
        self.mode=mode
        self.attention=SA_Module(input_nc=8)
        self.disp_range=disp_range
        self.conv=conv2d(3,32)
        self.occlusion_pred=Occlution_pred()
        self.occlusion_residual = Occlusion_residual(disp_range+1, deform=True)
        self.error_residual = Aggregation2D(disp_range)
    def forward(self, low_disp, left_img, right_img,idx=None):
        """
        left_img   B, C, H ,W
        low_disp   B, H, W or B. 1/2H, 1/2W
        """
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp * scale_factor  # scale correspondingly
        left_feature=self.conv(left_img)
        right_feature=self.conv(right_img)
        res_volume = res_dynamic_cost_volume(self.disp_range, left_feature, right_feature, disp, self.mode)

        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error_map = torch.abs(warped_right - left_img)  # [B, C, H, W]
        occlusion_input = torch.cat((error_map, right_img, disp), dim=1)
        occlusion_mask = self.occlusion_pred(occlusion_input)

        occluded_feature = torch.cat((res_volume, occlusion_mask), dim=1)
        occ_attention=self.attention(torch.cat((error_map, left_img, disp, occlusion_mask), dim=1))
        occlusion_residual = self.occlusion_residual(occluded_feature*occ_attention)

        error_feature=res_volume
        error_residual = self.error_residual( error_feature,None)



        disp = F.relu(disp + error_residual+occlusion_residual, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]
        
        if idx:
            if idx%10==1:
                vutils.save_image(occ_attention[0], "show/"+str(idx)+"_"+str(disp.size()[-1])+'occ_att.jpg', normalize=True)
                vutils.save_image(error_residual[0], "show/"+str(idx)+"_"+str(disp.size()[-1])+'error_r.jpg', normalize=True)
                

        return disp,occlusion_mask

class StereoDRNetRefinement(nn.Module):
    def __init__(self,attention):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        if attention:
            in_channels = 10
        else :
            in_channels=6
        self.attention=attention
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity
        self.attentionnet = SA_Module(input_nc=10)
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]

        if self.attention:
            query = torch.cat((left_img, right_img, error, disp), dim=1)
            attention_map=self.attentionnet(query)
           # concat = torch.cat((error, left_img), dim=1)  # [B, 4, H, W]
            conv1 = self.conv1(query*attention_map)
        else:
            concat = torch.cat((error, left_img), dim=1)  # [B, 4, H, W]
            conv1 = self.conv1(concat)

        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp


class HourglassRefinement(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(HourglassRefinement, self).__init__()

        # Left and warped error
        in_channels = 6
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = DeformConv2d(32, 32)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
        self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96, mdconv=True)
        self.conv4b = Conv2x(96, 128, mdconv=True)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):

        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp
