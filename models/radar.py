from models.bts import upconv
import torch
import torch.nn as nn
import torchvision.models as models

class encoder_radar_sparse_conv(nn.Module):
    def __init__(self, params):
        # radar encoder for the first stage
        super(encoder_radar_sparse_conv, self).__init__()

        self.params = params
        self.sparse_conv1 = SparseConv(params.radar_input_channels, 16, 7, activation='elu')
        self.sparse_conv2 = SparseConv(16, 16, 5, activation='elu')
        self.sparse_conv3 = SparseConv(16, 16, 3, activation='elu')
        self.sparse_conv4 = SparseConv(16, 3, 3, activation='elu')

        if params.encoder_radar == 'resnet34':
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder_radar == 'resnet18':
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        mask = (x[:, 0] > 0).float().unsqueeze(1)
        feature = x
        feature, mask = self.sparse_conv1(feature, mask)
        feature, mask = self.sparse_conv2(feature, mask)
        feature, mask = self.sparse_conv3(feature, mask)
        feature, mask = self.sparse_conv4(feature, mask)

        skip_feat = []
        i = 1
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat

class encoder_radar_sub(nn.Module):
    def __init__(self, params):
        # radar encoder for the second stage
        super(encoder_radar_sub, self).__init__()

        self.params = params
        import torchvision.models as models
        self.conv = torch.nn.Sequential(nn.Conv2d(params.radar_input_channels+1, 3, 3, 1, 1, bias=False),
                                        nn.ELU())

        if params.encoder_radar == 'resnet34':
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder_radar == 'resnet18':
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))
    def forward(self, x):
        feature = x
        feature = self.conv(feature)
        skip_feat = []
        i = 1
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat


class decoder_radar(nn.Module):
    def __init__(self, params, feat_out_channels_img, feat_out_channels_radar):
        super(decoder_radar, self).__init__()
        self.params     = params
        self.upconv5    = upconv(feat_out_channels_img[4]+feat_out_channels_radar[4], feat_out_channels_radar[4]//2)
        self.bn5        = nn.BatchNorm2d(feat_out_channels_radar[4]//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv5      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[4]//2, feat_out_channels_radar[4]//2, 3, 1, 1, bias=False),
                                              nn.ELU())
                                
        self.upconv4    = upconv(feat_out_channels_img[3]+feat_out_channels_radar[3]+feat_out_channels_radar[4]//2, feat_out_channels_radar[3]//2)
        self.bn4        = nn.BatchNorm2d(feat_out_channels_radar[3]//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[3]//2, feat_out_channels_radar[3]//2, 3, 1, 1, bias=False),
                                              nn.ELU())

        self.upconv3    = upconv(feat_out_channels_img[2]+feat_out_channels_radar[2]+feat_out_channels_radar[3]//2, feat_out_channels_radar[2]//2)
        self.bn3        = nn.BatchNorm2d(feat_out_channels_radar[2]//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[2]//2, feat_out_channels_radar[2]//2, 3, 1, 1, bias=False),
                                              nn.ELU())
                                        
        self.upconv2    = upconv(feat_out_channels_img[1]+feat_out_channels_radar[1]+feat_out_channels_radar[2]//2, feat_out_channels_radar[1]//2)
        self.bn2        = nn.BatchNorm2d(feat_out_channels_radar[1]//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[1]//2, feat_out_channels_radar[1]//2, 3, 1, 1, bias=False),
                                              nn.ELU())

        self.upconv1    = upconv(feat_out_channels_img[0]+feat_out_channels_radar[0]+feat_out_channels_radar[1]//2, feat_out_channels_radar[0]//2)
        self.bn1        = nn.BatchNorm2d(feat_out_channels_radar[0]//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv1      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[0]//2, feat_out_channels_radar[0]//2, 3, 1, 1, bias=False),
                                              nn.ELU())
                                            
        # self.get_depth  = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[0]//2, 1, 3, 1, 1, bias=False),
        #                                       nn.Sigmoid())
        
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[0]//2, 2, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, image_features, radar_features):
        img_skip0, img_skip1, img_skip2, img_skip3, img_final = image_features[0], image_features[1], image_features[2], image_features[3], image_features[4]
        rad_skip0, rad_skip1, rad_skip2, rad_skip3, rad_final = radar_features[0], radar_features[1], radar_features[2], radar_features[3], radar_features[4]
        final = torch.cat([img_final, rad_final], axis=1)
        upconv5 = self.upconv5(final)
        upconv5 = self.bn5(upconv5)
        upconv5 = self.conv5(upconv5)
        upconv5 = torch.cat([img_skip3, rad_skip3, upconv5], axis=1)

        upconv4 = self.upconv4(upconv5)
        upconv4 = self.bn4(upconv4)
        upconv4 = self.conv4(upconv4)
        upconv4 = torch.cat([img_skip2, rad_skip2, upconv4], axis=1)

        upconv3 = self.upconv3(upconv4)
        upconv3 = self.bn3(upconv3)
        upconv3 = self.conv3(upconv3)
        upconv3 = torch.cat([img_skip1, rad_skip1, upconv3], axis=1)

        upconv2 = self.upconv2(upconv3)
        upconv2 = self.bn2(upconv2)
        upconv2 = self.conv2(upconv2)
        upconv2 = torch.cat([img_skip0, rad_skip0, upconv2], axis=1)

        upconv1 = self.upconv1(upconv2)
        upconv1 = self.bn1(upconv1)
        upconv1 = self.conv1(upconv1)

        # confidence = self.get_depth(upconv1)
        # depth = self.params.max_depth * confidence
        depth_conf = self.get_depth(upconv1)
        depth = self.params.max_depth * depth_conf[:, 0:1]
        confidence = depth_conf[:, 1:2]

        return confidence, depth


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation='relu'):
        super().__init__()

        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'elu':
            self.act = nn.ELU()

        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        

    def forward(self, x, mask):
        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.act(x)
        
        mask = self.max_pool(mask)

        return x, mask
