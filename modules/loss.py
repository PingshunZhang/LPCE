import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou as iou

EPS = 1e-8
    
class focal_loss(nn.Module):
    def __init__(self, alpha=[0.75,0.75,0.25,0.75,0.75], gamma=2, num_classes = 5, size_average=True):

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):

        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        
        preds_logsoft = F.log_softmax(preds, dim=1) 
        
        preds_softmax = torch.exp(preds_logsoft)    
   
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        alpha = alpha.gather(0,labels.view(-1))

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
    
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
       
        return loss

class MSELoss4_iter_cluster(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, output_graphs):
        loss_nodes = F.mse_loss(output_graphs.x, output_graphs.y[:, :2])


        loss_edges = []
        n_modules_trn = output_graphs.edge_attr.shape[1] // 2
        for i in range(n_modules_trn):
            loss_edges.append(F.mse_loss(output_graphs.edge_attr[:, i * 2:(i + 1) * 2], output_graphs.edge_y))


        output_graphs_list = output_graphs.to_data_list()
        bs = len(output_graphs_list)
        loss_overlap = torch.zeros(bs, requires_grad=True).to(loss_nodes.device)
        for i in range(bs):
            pred_pos_norm = output_graphs_list[i].x + output_graphs_list[i].y[:, 2:4]
            pred_left_top = pred_pos_norm - output_graphs_list[i].y[:, 4:] / 2
            pred_right_bottom = pred_pos_norm + output_graphs_list[i].y[:, 4:] / 2

            pred_bboxes = torch.cat((pred_left_top, pred_right_bottom), dim=-1)
            label_iou = iou(pred_bboxes, pred_bboxes)
            label_iou = label_iou - torch.diag(label_iou.diagonal())

            loss_overlap[i] = torch.sum(label_iou) / (pred_bboxes.shape[0] * pred_bboxes.shape[0] - pred_bboxes.shape[0] + EPS)

        loss = self.alpha * loss_nodes + self.beta * sum(loss_edges) / n_modules_trn + self.gamma * torch.mean(loss_overlap)


        return loss