import torch
import torch.nn as nn

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.l1_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.mse_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class binary_cross_entropy(nn.Module):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def forward(self, confidence, radar_gt, mask):
        loss = torch.nn.functional.binary_cross_entropy(confidence, radar_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class smoothness_loss_func(nn.Module):
    def __init__(self):
        super(smoothness_loss_func, self).__init__()
    
    def gradient_yx(self, T):
        '''
        Computes gradients in the y and x directions

        Arg(s):
            T : tensor
                N x C x H x W tensor
        Returns:
            tensor : gradients in y direction
            tensor : gradients in x direction
        '''

        dx = T[:, :, :, :-1] - T[:, :, :, 1:]
        dy = T[:, :, :-1, :] - T[:, :, 1:, :]
        return dy, dx
    
    def forward(self, predict, image):
        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
        
        return smoothness_x + smoothness_y

