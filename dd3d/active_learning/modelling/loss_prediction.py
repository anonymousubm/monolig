'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

class LossPrediction(nn.Module):
    def __init__(
        self,
        feature_sizes=[(48,160), (24,80), (12,40), (6,20), (3,10)],
        num_channels=[256, 256, 256, 256, 256],
        interm_dim=128
    ):
        super(LossPrediction, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])
        self.GAP5 = nn.AvgPool2d(feature_sizes[4])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.FC5 = nn.Linear(num_channels[4], interm_dim)

        self.linear = nn.Linear(5 * interm_dim, 1)
    
    def extract_features(self, data, model):
        interm_features = [[] for _ in range(5)]
        losses = []
        for single_data in data:
            single_loss_dict, features = model([single_data])        
            out0 = nn.AvgPool2d(features[0].shape[2:])(features[0].detach())
            out0 = out0.view(out0.size(0), -1)
            interm_features[0].append(out0)

            out1 = nn.AvgPool2d(features[1].shape[2:])(features[1].detach())
            out1 = out1.view(out1.size(0), -1)
            interm_features[1].append(out1)
            
            out2 = nn.AvgPool2d(features[2].shape[2:])(features[2].detach())
            out2 = out2.view(out2.size(0), -1)
            interm_features[2].append(out2)
            
            out3 = nn.AvgPool2d(features[3].shape[2:])(features[3].detach())
            out3 = out3.view(out3.size(0), -1)
            interm_features[3].append(out3)
            
            out4 = nn.AvgPool2d(features[4].shape[2:])(features[4].detach())
            out4 = out4.view(out4.size(0), -1)
            interm_features[4].append(out4)
            
            single_loss_dict = {
                name: loss.detach() for name, loss in single_loss_dict.items()
            }
            single_loss = sum(single_loss_dict.values()).detach()
            losses.append(single_loss)
        interm_features = [torch.cat(f) for f in interm_features]
        losses = torch.cat([l.view(1,1) for l in losses]).view(-1)
        return losses, interm_features

    def forward(self, data, model):
        target_loss, features = self.extract_features(data, model)
        
        out1 = F.relu(self.FC1(features[0]))
        out2 = F.relu(self.FC2(features[1]))
        out3 = F.relu(self.FC3(features[2]))
        out4 = F.relu(self.FC4(features[3]))
        out5 = F.relu(self.FC5(features[4]))
        pred_loss = self.linear(torch.cat((out1, out2, out3, out4, out5), 1))
        pred_loss = pred_loss.view(pred_loss.size(0))
        if self.training:
            loss_loss_pred = LossPredLoss(pred_loss, target_loss, margin=1.0)
            return loss_loss_pred
        return pred_loss

def build_loss_prediction_optimizer(model, lr=1e-3, momentum=0.9, weight_decay=5e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                                momentum=momentum, weight_decay=weight_decay)
    return optimizer