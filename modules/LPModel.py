import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import ModuleList, Linear, LayerNorm
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter
from torchvision.ops import roi_align
import os

class LPModel(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            node_input_dim: int,
            edge_input_dim: int,
            node_dim: int,
            edge_dim: int,
            node_hid_dim: int,
            edge_hid_dim: int,
            num_experts: int,
            expert_hid_dim: int,
            output_dim: int,
            disable_edge_updates: bool,
            train_fe: bool,
            normalization: bool,
            backbone: str
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.disable_edge_updates = disable_edge_updates
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hid_dim = node_hid_dim
        self.edge_hid_dim = edge_hid_dim
        self.num_experts = num_experts
        self.expert_hid_dim = expert_hid_dim
        self.output_dim = output_dim

        self.train_fe = train_fe
        self.normalization = normalization
        self.backbone = backbone

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.feat_ext = FeatureExtraction(self.train_fe, self.normalization, self.backbone)

        self.lp_gnns = LPGNN(self.n_layers,
                                         self.n_heads,
                                         self.node_input_dim,
                                         self.edge_input_dim,
                                         self.node_dim,
                                         self.edge_dim,
                                         self.node_hid_dim,
                                         self.edge_hid_dim,
                                         self.num_experts,
                                         self.expert_hid_dim,
                                         self.output_dim,
                                         self.disable_edge_updates,
                                         self.backbone)

    def build_edge_point_idx2(self, edge_index, num_samples, node_bid, device):
        tails = edge_index[0]
        heads = edge_index[1]
        M = edge_index.shape[1]

        edge_point_idx_global = []
        generic_edge_point_idx = []
        self_edge_point_idx = []
        for i in range(M):
            if tails[i] != heads[i]:
                edge_point_idx_global.extend([i] * num_samples)
                generic_edge_point_idx.extend([len(edge_point_idx_global) - num_samples + j for j in range(num_samples)])
            else:
                edge_point_idx_global.append(i)
                self_edge_point_idx.append(len(edge_point_idx_global) - 1)

        edge_point_bid = node_bid[tails][edge_point_idx_global]
        edge_point_idx_global = torch.tensor(edge_point_idx_global).to(device)

        generic_edge_index = torch.where(tails != heads)[0].tolist()
        generic_edge_tails = tails[generic_edge_index]
        generic_edge_heads = heads[generic_edge_index]

        self_edge_index = torch.where(tails == heads)[0].tolist()
        self_edge_tails = tails[self_edge_index]

        return (edge_point_idx_global,
                edge_point_bid,
                generic_edge_point_idx,
                self_edge_point_idx,
                generic_edge_tails,
                generic_edge_heads,
                self_edge_tails)

    def build_node_patch(self, pre_label_pos_resize, patch_size):
        patch_left_top = pre_label_pos_resize - patch_size / 2
        patch_right_bottom = pre_label_pos_resize + patch_size / 2
        node_patch = torch.cat([patch_left_top, patch_right_bottom], dim=-1)

        return node_patch
    def build_edge_patch2(self,
                          pre_label_pos_resize,
                          patch_size,
                          num_samples,
                          node_patch,
                          generic_edge_point_idx,
                          self_edge_point_idx,
                          generic_edge_tails,
                          generic_edge_heads,
                          self_edge_tails,
                          device):
        generic_edge_cor_tail = pre_label_pos_resize[generic_edge_tails]
        generic_edge_cor_head = pre_label_pos_resize[generic_edge_heads]

        t_values = torch.linspace(0, 1, num_samples).to(device).view(-1, 1)
        generic_edge_points = (1 - t_values) * generic_edge_cor_tail.unsqueeze(1) + t_values * generic_edge_cor_head.unsqueeze(1)

        left_top = generic_edge_points - patch_size / 2
        right_bottom = generic_edge_points + patch_size / 2

        generic_edge_patches = torch.cat([left_top, right_bottom], dim=-1).view(-1, 4)

        self_edge_patches = node_patch[self_edge_tails]

        edge_patches = torch.zeros((len(self_edge_point_idx) + len(generic_edge_point_idx), 4)).to(device)
        edge_patches[self_edge_point_idx] = self_edge_patches
        edge_patches[generic_edge_point_idx] = generic_edge_patches

        return edge_patches
    def feat_align(self, img_feat, batch_id, patch):
        patch_feat = roi_align(img_feat,
                               torch.cat([batch_id.unsqueeze(1), patch], dim=-1),
                               output_size=(2, 2),
                               spatial_scale=img_feat.shape[-1] / 256,
                               sampling_ratio=2,
                               aligned=True)
        patch_feat = self.pooling(patch_feat).squeeze(dim=(2, 3))

        return patch_feat
    def build_vis_feat(self, img_feat1, img_feat2, batch_id, patch):
        patch_feat1 = self.feat_align(img_feat1, batch_id, patch)
        patch_feat2 = self.feat_align(img_feat2, batch_id, patch)

        if self.normalization:
            patch_feat1 = F.normalize(patch_feat1, p=2, dim=1)
            patch_feat2 = F.normalize(patch_feat2, p=2, dim=1)

        patch_feat = torch.cat([patch_feat1, patch_feat2], dim=-1)

        return patch_feat
    
    
    def forward(self, graph, images, file_name):
        device = images.device
        node_bid = graph.batch
        edge_index = graph.edge_index
        num_samples = 7


        (edge_point_idx_global,
         edge_point_bid,
         generic_edge_point_idx,
         self_edge_point_idx,
         generic_edge_tails,
         generic_edge_heads,
         self_edge_tails) = self.build_edge_point_idx2(edge_index, num_samples, node_bid, device)

        patch_size = 5

        node_tensors1 = graph.x
        edge_tensors1 = graph.edge_attr

        img_feat1, img_feat2 = self.feat_ext(images)

        pre_label_pos_resize = node_tensors1[:, :2] * 256
        node_patch = self.build_node_patch(pre_label_pos_resize, patch_size)

        edge_patch = self.build_edge_patch2(pre_label_pos_resize,
                                            patch_size,
                                            num_samples,
                                            node_patch,
                                            generic_edge_point_idx,
                                            self_edge_point_idx,
                                            generic_edge_tails,
                                            generic_edge_heads,
                                            self_edge_tails,
                                            device)

        node_tensors2 = self.build_vis_feat(img_feat1, img_feat2, node_bid, node_patch)
        edge_tensors2 = self.build_vis_feat(img_feat1, img_feat2, edge_point_bid, edge_patch)
        edge_tensors2 = scatter(edge_tensors2, edge_point_idx_global, dim=0, reduce='mean')

        out_nodes, out_edges = self.lp_gnns(node_tensors1, node_tensors2, edge_tensors1, edge_tensors2, edge_index, node_bid, file_name)

        graph.x = out_nodes
        graph.edge_attr = out_edges
        
        return graph


class FeatureExtraction(nn.Module):
    def __init__(self, train_fe=False, normalization=True, backbone='resnet101'):
        super().__init__()
        self.normalization = normalization
        if backbone == 'resnet101':
            backbone_model = models.resnet101(weights='IMAGENET1K_V1')
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            layer3 = 'layer3'
            layer4 = 'layer4'
            layer3_idx = resnet_feature_layers.index(layer3)
            layer4_idx = resnet_feature_layers.index(layer4)
            resnet_module_list = [backbone_model.conv1,
                                  backbone_model.bn1,
                                  backbone_model.relu,
                                  backbone_model.maxpool,
                                  backbone_model.layer1,
                                  backbone_model.layer2,
                                  backbone_model.layer3,
                                  backbone_model.layer4]
            self.feat_ext1 = nn.Sequential(*resnet_module_list[:layer3_idx + 1])
            self.feat_ext2 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        if backbone == 'vgg16':
            self.model = models.vgg16_bn(weights='IMAGENET1K_V1')
            conv_layers = nn.Sequential(*list(self.model.features.children()))
            conv_list = feat1_list = feat2_list = []

            cnt_m, cnt_r = 1, 0
            for layer, module in enumerate(conv_layers):
                if isinstance(module, nn.Conv2d):
                    cnt_r += 1
                if isinstance(module, nn.MaxPool2d):
                    cnt_r = 0
                    cnt_m += 1
                conv_list += [module]

                if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                    feat1_list = conv_list
                    conv_list = []
                elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
                    feat2_list = conv_list
                    conv_list = []

            assert len(feat1_list) > 0 and len(feat2_list) > 0

            self.feat_ext1 = nn.Sequential(*feat1_list)
            self.feat_ext2 = nn.Sequential(*feat2_list)

        if not train_fe:
            for param in self.feat_ext1.parameters():
                param.requires_grad = False
            for param in self.feat_ext2.parameters():
                param.requires_grad = False

    def forward(self, image_batch):
        image_feat1 = self.feat_ext1(image_batch)
        image_feat2 = self.feat_ext2(image_feat1)

        return image_feat1, image_feat2


class LPGNN(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            node_input_dim: int,
            edge_input_dim: int,
            node_dim: int,
            edge_dim: int,
            node_hid_dim: int,
            edge_hid_dim: int,
            num_experts: int,
            expert_hid_dim: int,
            output_dim: int,
            disable_edge_updates: bool,
            backbone: str
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.disable_edge_updates = disable_edge_updates
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hid_dim = node_hid_dim
        self.edge_hid_dim = edge_hid_dim
        self.num_experts = num_experts
        self.expert_hid_dim = expert_hid_dim
        self.output_dim = output_dim
        self.backbone = backbone

        self.node_enc1 = Linear(self.node_input_dim, self.node_dim)
        self.edge_enc1 = Linear(self.edge_input_dim, self.edge_dim)

        if self.backbone == 'resnet101':
            self.node_enc2 = Linear(3072, self.node_dim)
            self.edge_enc2 = Linear(3072, self.edge_dim)
        elif self.backbone == 'vgg16':
            self.node_enc2 = Linear(1024, self.node_dim)
            self.edge_enc2 = Linear(1024, self.edge_dim)

        self.layers = ModuleList([LPGNNLayer(self.n_heads,
                                             self.node_dim,
                                             self.edge_dim,
                                             self.node_hid_dim,
                                             self.edge_hid_dim,
                                             self.disable_edge_updates) for _ in range(self.n_layers - 1)])

        self.moe = MoE(self.n_heads,
                        self.node_dim,
                        self.edge_dim,
                        self.node_hid_dim,
                        self.edge_hid_dim,
                        self.num_experts,
                        self.expert_hid_dim,
                        self.disable_edge_updates)

        self.RD = RD(self.n_heads, self.node_dim)
        self.ln_cross = LayerNorm(self.node_dim)
        self.node_ln = LayerNorm(self.node_dim)

        self.edge_RD = RD(self.n_heads, self.node_dim)
        self.edge_ln_cross = LayerNorm(self.node_dim)
        self.edge_ln = LayerNorm(self.node_dim)
        
        self.node_dec = Linear(self.node_dim, self.output_dim)
        self.edge_dec = Linear(self.edge_dim, self.output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.node_dec.weight)
        nn.init.zeros_(self.node_dec.bias)
        nn.init.zeros_(self.edge_dec.weight)
        nn.init.zeros_(self.edge_dec.bias)

    def find_three_tensors(self, file_name):
        output_dir = 'data/downloaded/three_style_features'
        three_tensors = torch.load(os.path.join(output_dir, f"{file_name[:5]}.pt"))
        return three_tensors

    def forward(self, node_tensors1, node_tensors2, edge_tensors1, edge_tensors2, edge_index, node_bid, file_name):


        node_tensors = self.node_enc1(node_tensors1) + self.node_enc2(node_tensors2)
        edge_tensors = self.edge_enc1(edge_tensors1) + self.edge_enc2(edge_tensors2)

        device = node_tensors.device
        tensor_knns = []
        for fn in file_name:
            tensor_one_knns = self.find_three_tensors(fn).to(device)
            tensor_knns.append(tensor_one_knns)

        counts = torch.bincount(node_bid)
        indices = torch.cumsum(counts, dim=0)
        indices = torch.cat((torch.tensor([0], device=device), indices[:-1]))
        rd_tensors = []
        for i, (start, count) in enumerate(zip(indices, counts)):
            end = start + count
            rd_tensors.append(self.RD(node_tensors[start:end, :], tensor_knns[i]))
        node_tensors = self.node_ln(node_tensors) + self.ln_cross(torch.cat(rd_tensors, dim=0))
        
        edge_counts = torch.bincount(node_bid) * torch.bincount(node_bid)
        edge_indices = torch.cumsum(edge_counts, dim=0)
        edge_indices = torch.cat((torch.tensor([0], device=device), edge_indices[:-1]))
        edge_rd_tensors = []
        for i, (edge_start, edge_count) in enumerate(zip(edge_indices, edge_counts)):
            edge_end = edge_start + edge_count
            edge_rd_tensors.append(self.edge_RD(edge_tensors[edge_start:edge_end, :], tensor_knns[i]))
        edge_tensors = self.edge_ln(edge_tensors) + self.edge_ln_cross(torch.cat(edge_rd_tensors, dim=0))

        for layer_id in range(self.n_layers-1):
            node_tensors, edge_tensors = self.layers[layer_id](node_tensors, edge_tensors, edge_index)

        node_tensors, edge_tensors = self.moe(node_tensors, edge_tensors, node_bid, edge_index)

        out_nodes = self.node_dec(node_tensors)
        out_edges = self.edge_dec(edge_tensors)

        return out_nodes, out_edges
    
class RD(nn.Module):
    def __init__(self,
                 n_heads: int,
                 node_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.Wnq = Linear(self.node_dim, self.node_dim)
        self.Wnk = Linear(self.node_dim, self.node_dim)
        self.Wnv = Linear(self.node_dim, self.node_dim)

        self.out_proj = Linear(self.node_dim, self.node_dim)

    def separate_heads(self, x):
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        x = x.contiguous().view(new_shape)
        return x.transpose(0, 1)

    def concatenate_heads(self, x):
        x = x.permute(1, 0, 2)
        new_shape = x.shape[:-2] + (self.node_dim,)

        return x.contiguous().view(new_shape)

    def forward(self, node_tensors, tensor_knn):
        Q = self.Wnq(node_tensors)
        K = self.Wnk(tensor_knn)
        V = self.Wnv(tensor_knn)

        Q = self.separate_heads(Q)
        K = self.separate_heads(K)
        V = self.separate_heads(V)

        attn_score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weight = F.softmax(attn_score, dim=-1)
        update_node_tensors = torch.matmul(attn_weight,V)

        update_node_tensors = self.out_proj(self.concatenate_heads(update_node_tensors))

        return update_node_tensors

class MoE(nn.Module):
    def __init__(
            self,
            n_heads: int,
            node_dim: int,
            edge_dim: int,
            node_hid_dim: int,
            edge_hid_dim: int,
            num_experts: int,
            expert_hid_dim: int,
            disable_edge_updates: bool,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.disable_edge_updates = disable_edge_updates
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hid_dim = node_hid_dim
        self.edge_hid_dim = edge_hid_dim
        self.num_experts = num_experts
        self.expert_hid_dim = expert_hid_dim


        self.node_attn_ln1 = LayerNorm(self.node_dim)
        self.edge_attn_ln1 = LayerNorm(self.edge_dim)
        self.node_mha = NodeMultiHeadAttention(self.n_heads, self.node_dim, self.edge_dim)

        self.node_ffn_ln = LayerNorm(self.node_dim)
        self.node_ffn_fc1 = Linear(self.node_dim, self.node_hid_dim)
        self.node_ffn_fc2 = Linear(self.node_hid_dim, self.node_dim)
        self.experts = nn.ModuleList([FFN(self.node_dim,self.expert_hid_dim) for _ in range(self.num_experts)])
        self.edge_experts = nn.ModuleList([FFN(self.edge_dim,self.expert_hid_dim) for _ in range(self.num_experts)])

        self.moe_node_ffn_ln = LayerNorm(self.node_dim)
        self.moe_node_ffn_fc1 = Linear(self.node_dim, self.node_hid_dim)
        self.moe_node_ffn_fc2 = Linear(self.node_hid_dim, self.node_dim)

        self.moe_edge_ffn_ln = LayerNorm(self.edge_dim)
        self.moe_edge_ffn_fc1 = Linear(self.edge_dim, self.edge_hid_dim)
        self.moe_edge_ffn_fc2 = Linear(self.edge_hid_dim, self.edge_dim)

        self.node_attn_ln2 = LayerNorm(self.node_dim)
        self.edge_attn_ln2 = LayerNorm(self.edge_dim)
        self.edge_mha = EdgeMultiHeadAttention(self.n_heads, self.node_dim, self.edge_dim)

        self.edge_ffn_ln = LayerNorm(self.edge_dim)
        self.edge_ffn_fc1 = Linear(self.edge_dim, self.edge_hid_dim)
        self.edge_ffn_fc2 = Linear(self.edge_hid_dim, self.edge_dim)

        self.gate_dim = 300
        self.gate = gate(self.node_dim, self.gate_dim)
        self.cluster = nn.Parameter(torch.Tensor(self.num_experts, self.gate_dim))
        torch.nn.init.xavier_normal_(self.cluster.data)

    def get_q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster, 2), 2) )
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def init_weights(self):
        nn.init.zeros_(self.node_dec.weight)
        nn.init.zeros_(self.node_dec.bias)
        nn.init.zeros_(self.edge_dec.weight)
        nn.init.zeros_(self.edge_dec.bias)
    
    def forward(self, node_tensors, edge_tensors, node_bid, edge_index):
        node_tensors_prime = self.node_mha(self.node_attn_ln1(node_tensors),
                                           self.edge_attn_ln1(edge_tensors),
                                           edge_index) + node_tensors

        node_tensors_new = self.node_ffn_fc2(
            F.relu(self.node_ffn_fc1(self.node_ffn_ln(node_tensors_prime)))) + node_tensors_prime

        gate_input = global_add_pool(node_tensors_new, node_bid)
        z = self.gate(gate_input) 
        q = self.get_q(z)
        scores = F.gumbel_softmax(q, hard=True, dim=-1)

        counts = torch.bincount(node_bid)
        all_node_scores = torch.repeat_interleave(scores, counts, dim=0).unsqueeze(2)

        edge_counts = counts * counts
        all_edge_scores = torch.repeat_interleave(scores, edge_counts, dim=0).unsqueeze(2)

        all_node_tensors = torch.stack([expert(node_tensors_new) for expert in self.experts], dim = 1)
        moe_node_tensors = (all_node_scores * all_node_tensors).sum(dim=1)

        node_tensors_new = node_tensors_new + moe_node_tensors

        node_tensors_new = self.moe_node_ffn_fc2(
            F.relu(self.moe_node_ffn_fc1(self.moe_node_ffn_ln(node_tensors_new)))) + node_tensors_new

        edge_tensors_prime = self.edge_mha(self.node_attn_ln2(node_tensors_new),
                                           self.edge_attn_ln2(edge_tensors),
                                           edge_index) + edge_tensors
        edge_tensors_new = self.edge_ffn_fc2(
            F.relu(self.edge_ffn_fc1(self.edge_ffn_ln(edge_tensors_prime)))) + edge_tensors_prime
        
        all_edge_tensors = torch.stack([edge_expert(edge_tensors_new) for edge_expert in self.edge_experts], dim = 1)
        moe_edge_tensors = (all_edge_scores * all_edge_tensors).sum(dim=1)

        edge_tensors_new = edge_tensors_new + moe_edge_tensors

        edge_tensors_new = self.moe_edge_ffn_fc2(
            F.relu(self.moe_edge_ffn_fc1(self.moe_edge_ffn_ln(edge_tensors_new)))) + edge_tensors_new

        return node_tensors_new, edge_tensors_new

class gate(torch.nn.Module):
    def __init__(self, emb_dim, gate_dim=300):
        super(gate, self).__init__()
        self.linear1 = nn.Linear(emb_dim, gate_dim)
        self.batchnorm = nn.BatchNorm1d(gate_dim)
        self.linear2 = nn.Linear(gate_dim, gate_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        gate_emb = self.linear2(x)
        return gate_emb

class FFN(nn.Module):
    def __init__(
            self,
            node_dim: int,
            node_hid_dim: int
    ):
        super().__init__()
        self.node_ffn_fc1 = Linear(node_dim, node_hid_dim)
        self.node_ffn_fc2 = Linear(node_hid_dim, node_dim)
        self.ln = LayerNorm(node_dim)

    def forward(self, x):
        x = self.node_ffn_fc2(
            F.relu(self.node_ffn_fc1(self.ln(x))))
        return x
    
class LPGNNLayer(nn.Module):
    def __init__(
            self,
            n_heads: int,
            node_dim: int,
            edge_dim: int,
            node_hid_dim: int,
            edge_hid_dim: int,
            disable_edge_updates: bool
    ):
        super().__init__()
        self.n_heads = n_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hid_dim = node_hid_dim
        self.edge_hid_dim = edge_hid_dim
        self.disable_edge_updates = disable_edge_updates

        self.node_attn_ln1 = LayerNorm(self.node_dim)
        self.edge_attn_ln1 = LayerNorm(self.edge_dim)
        self.node_mha = NodeMultiHeadAttention(self.n_heads, self.node_dim, self.edge_dim)

        self.node_ffn_ln = LayerNorm(self.node_dim)
        self.node_ffn_fc1 = Linear(self.node_dim, self.node_hid_dim)
        self.node_ffn_fc2 = Linear(self.node_hid_dim, self.node_dim)

        self.node_attn_ln2 = LayerNorm(self.node_dim)
        self.edge_attn_ln2 = LayerNorm(self.edge_dim)
        self.edge_mha = EdgeMultiHeadAttention(self.n_heads, self.node_dim, self.edge_dim)

        self.edge_ffn_ln = LayerNorm(self.edge_dim)
        self.edge_ffn_fc1 = Linear(self.edge_dim, self.edge_hid_dim)
        self.edge_ffn_fc2 = Linear(self.edge_hid_dim, self.edge_dim)

    def forward(self, node_tensors, edge_tensors, edge_index):
        node_tensors_prime = self.node_mha(self.node_attn_ln1(node_tensors),
                                           self.edge_attn_ln1(edge_tensors),
                                           edge_index) + node_tensors

        node_tensors_new = self.node_ffn_fc2(
            F.relu(self.node_ffn_fc1(self.node_ffn_ln(node_tensors_prime)))) + node_tensors_prime

        edge_tensors_prime = self.edge_mha(self.node_attn_ln2(node_tensors_new),
                                           self.edge_attn_ln2(edge_tensors),
                                           edge_index) + edge_tensors

        edge_tensors_new = self.edge_ffn_fc2(
            F.relu(self.edge_ffn_fc1(self.edge_ffn_ln(edge_tensors_prime)))) + edge_tensors_prime

        return node_tensors_new, edge_tensors_new


class NodeMultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads: int,
                 node_dim: int,
                 edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.edge_dim = edge_dim

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.Wnq = Linear(self.node_dim, self.node_dim)
        self.Wnk = Linear(self.node_dim, self.node_dim)
        self.Wnv = Linear(self.node_dim, self.node_dim)

        self.Weq = Linear(self.edge_dim, self.node_dim)
        self.Wek = Linear(self.edge_dim, self.node_dim)
        self.Wev = Linear(self.edge_dim, self.node_dim)

        self.out_proj = Linear(self.node_dim, self.node_dim)

    def separate_heads(self, x):
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        x = x.contiguous().view(new_shape)

        return x.transpose(0, 1)

    def concatenate_heads(self, x):
        x = x.permute(1, 0, 2)
        new_shape = x.shape[:-2] + (self.node_dim,)

        return x.contiguous().view(new_shape)

    def forward(self, node_tensors, edge_tensors, edge_index):
        eQ = self.Weq(edge_tensors)
        eK = self.Wek(edge_tensors)
        eV = self.Wev(edge_tensors)

        nQ = self.Wnq(node_tensors)
        nK = self.Wnk(node_tensors)
        nV = self.Wnv(node_tensors)

        eQ = self.separate_heads(eQ)
        eK = self.separate_heads(eK)
        eV = self.separate_heads(eV)

        nQ = self.separate_heads(nQ)
        nK = self.separate_heads(nK)
        nV = self.separate_heads(nV)

        Q = eQ + nQ[:, edge_index[0, :], :]
        K = eK + nK[:, edge_index[1, :], :]
        attn_score = torch.mul(Q, K).sum(dim=-1) * self.scale
        attn_weight = softmax(attn_score, edge_index[0, :], dim=-1)

        V = eV + nV[:, edge_index[1, :], :]
        update_node_tensors = scatter(torch.mul(attn_weight.unsqueeze(-1), V), edge_index[0, :], dim=1, reduce='sum')

        update_node_tensors = self.out_proj(self.concatenate_heads(update_node_tensors))

        return update_node_tensors


class EdgeMultiHeadAttention(nn.Module):
    def __init__(self,
                 n_heads: int,
                 node_dim: int,
                 edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = edge_dim // n_heads
        self.edge_dim = edge_dim

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.Wnq = Linear(self.node_dim, self.edge_dim)
        self.Wnk = Linear(self.node_dim, self.edge_dim)
        self.Wnv = Linear(self.node_dim, self.edge_dim)

        self.Weq = Linear(self.edge_dim, self.edge_dim)
        self.Wek = Linear(self.edge_dim, self.edge_dim)
        self.Wev = Linear(self.edge_dim, self.edge_dim)

        self.out_proj = Linear(self.edge_dim, self.edge_dim)

    def separate_heads(self, x):
        new_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        x = x.contiguous().view(new_shape)

        return x.transpose(0, 1)

    def concatenate_heads(self, x):
        x = x.permute(1, 0, 2)
        new_shape = x.shape[:-2] + (self.node_dim,)

        return x.contiguous().view(new_shape)

    def forward(self, node_tensors, edge_tensors, edge_index):
        eQ = self.Weq(edge_tensors)
        eK = self.Wek(edge_tensors)
        eV = self.Wev(edge_tensors)

        nQ = self.Wnq(node_tensors)
        nK = self.Wnk(node_tensors)
        nV = self.Wnv(node_tensors)

        eQ = self.separate_heads(eQ)
        eK = self.separate_heads(eK)
        eV = self.separate_heads(eV)

        nQ = self.separate_heads(nQ)
        nK = self.separate_heads(nK)
        nV = self.separate_heads(nV)

        N = node_tensors.shape[0]
        M = edge_tensors.shape[0]

        edge_node_incidence = torch.zeros(M, N).to(next(self.parameters()).device)
        edge_node_incidence[torch.arange(M), edge_index[0]] = 1
        edge_node_incidence[torch.arange(M), edge_index[1]] = 1

        node_edge_incidence = edge_node_incidence.t()

        edge_neighbor = torch.mm(edge_node_incidence, node_edge_incidence)
        neighbor_edge_index = torch.nonzero(edge_neighbor)

        Q = eQ[:, neighbor_edge_index[:, 0], :] + \
            nQ[:, edge_index[0, neighbor_edge_index[:, 0]], :] + \
            nQ[:, edge_index[1, neighbor_edge_index[:, 0]], :]
        K = eK[:, neighbor_edge_index[:, 1], :] + \
            nK[:, edge_index[0, neighbor_edge_index[:, 1]], :] + \
            nK[:, edge_index[1, neighbor_edge_index[:, 1]], :]
        attn_score = torch.mul(Q, K).sum(dim=-1) * self.scale
        attn_weight = softmax(attn_score, neighbor_edge_index[:, 0], dim=-1)

        V = eV[:, neighbor_edge_index[:, 1], :] + \
            nV[:, edge_index[0, neighbor_edge_index[:, 1]], :] + \
            nV[:, edge_index[1, neighbor_edge_index[:, 1]], :]
        update_edge_tensors = scatter(torch.mul(attn_weight.unsqueeze(-1), V),
                                      neighbor_edge_index[:, 0], dim=1, reduce='sum')

        update_edge_tensors = self.out_proj(self.concatenate_heads(update_edge_tensors))

        return update_edge_tensors