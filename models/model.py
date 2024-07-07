import torch
import torch.nn as nn
from models.bts import encoder_image, bts_gated_fuse
from models.radar import encoder_radar_sparse_conv, encoder_radar_sub, decoder_radar

class CaFNet(nn.Module):
    def __init__(self, params, threshold=0.4):
        super(CaFNet, self).__init__()
        self.threshold = threshold
        self.encoder = encoder_image(params)
        self.encoder_radar1 = encoder_radar_sparse_conv(params)
        self.decoder_radar = decoder_radar(params, self.encoder.feat_out_channels, self.encoder_radar1.feat_out_channels)
        self.encoder_radar2 = encoder_radar_sub(params)
        self.decoder = bts_gated_fuse(params, self.encoder.feat_out_channels, self.encoder_radar2.feat_out_channels, params.bts_size)


    def forward(self, x, radar, focal):

        skip_feat = self.encoder(x)
        skip_feat_radar = self.encoder_radar1(radar)
        rad_confidence, rad_depth = self.decoder_radar(skip_feat, skip_feat_radar)
        mask = (rad_confidence > self.threshold).float()
        radar_new_input = torch.cat([mask*rad_depth, radar], axis=1)
        skip_feat_radar_new = self.encoder_radar2(radar_new_input)
        
        depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.decoder(skip_feat, skip_feat_radar_new, focal, rad_confidence)

        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth, rad_confidence, rad_depth
