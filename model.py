# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool, TemporalEncoding, GINConv, GCNConv, BatchNorm, \
    global_max_pool, LayerNorm, SAGEConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dropout_edge, mask_feature, \
    to_networkx
from torch_geometric.data import Data, Batch
import numpy as np
import math
import networkx as nx
import copy
from load_data import text_to_vector
from functools import partial


class textprompt(nn.Module):
    def __init__(self, in_dim):
        super(textprompt, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features=in_dim, out_features=128),
                                 nn.Tanh(),
                                 nn.Linear(in_features=128, out_features=in_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.prompttype = 'add'
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h):

        h = self.dropout(h)
        if self.prompttype == 'add':
            weight = self.weight.repeat(h.shape[0], 1)
            h = self.mlp(h) + h
        if self.prompttype == 'mul':
            h = self.mlp(h) * h

        return h


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TDrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

        self.ln1 = LayerNorm(128)
        self.ln2 = LayerNorm(out_feats)

    def forward(self, x, edge_index, batch):
        h_list = []
        h = self.conv1(x, edge_index)
        # h = self.ln1(h)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h_list.append(h)

        h = self.conv2(h, edge_index)
        # h = self.ln2(h)
        # h = F.dropout(h, training=self.training)
        h_list.append(h)

        # hs = global_add_pool(torch.cat(h_list, dim=1), batch)
        hs = global_add_pool(h, batch)

        return hs, h


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(BUrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

        self.ln1 = LayerNorm(128)
        self.ln2 = LayerNorm(out_feats)

    def forward(self, x, edge_index, batch):
        edge_index = torch.flip(edge_index, dims=[0])

        h_list = []
        h = self.conv1(x, edge_index)
        # h = self.ln1(h)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h_list.append(h)

        h = self.conv2(h, edge_index)
        # h = self.ln2(h)
        # h = F.dropout(h, training=self.training)
        h_list.append(h)

        # hs = global_add_pool(torch.cat(h_list, dim=1), batch)
        hs = global_add_pool(h, batch)

        return hs, h


def compute_one_level_ratios(data):
    edge_index = data.edge_index
    batch = data.batch
    num_graphs = batch.max().item() + 1

    total_non_source_nodes = 0
    s = []
    for i in range(num_graphs):
        node_mask = (batch == i)
        nodes = node_mask.nonzero(as_tuple=True)[0]

        source_node = nodes[0].item()
        current_nodes = set(nodes.tolist())

        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_edge_index = edge_index[:, mask]

        non_source_nodes = len(current_nodes)
        total_non_source_nodes += non_source_nodes

        one_level_count = (sub_edge_index[0] == source_node).sum().item()
        # total_one_level_nodes += one_level_count
        s.append(one_level_count / non_source_nodes)
    return s


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, out_feats, t, u):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)
        self.proj_head = nn.Sequential(nn.Linear((out_feats) * 2, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        self.t = t
        self.b = 0.1
        self.u = u
        self.dim = 128
        self.prompt = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=self.dim),
            LayerNorm(self.dim),
            nn.Tanh(),
            nn.Linear(in_features=self.dim, out_features=in_feats))
        
        self.prompt2 = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=self.dim),
            LayerNorm(self.dim),
            nn.Tanh(),
            nn.Linear(in_features=self.dim, out_features=in_feats))
        
        self.input_adapter = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_feats, in_feats),
            # LayerNorm(in_feats),
            nn.LeakyReLU()
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.prompt:
            if isinstance(layer, nn.Linear):
                if layer.out_features == self.dim:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    # nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.prompt2:
            if isinstance(layer, nn.Linear):
                if layer.out_features == self.dim:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    # nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def process_data(self, data):

        x = data.x
        x = self.input_adapter(x)
        root = (data.batch[1:] - data.batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        
        one_level = torch.FloatTensor(compute_one_level_ratios(data)).to(x.device)
        alpha_g = torch.sigmoid((one_level - self.u) / self.b)
        alpha = alpha_g[data.batch].unsqueeze(-1)
        
        # ---- prompts from root ----
        root_feat = x[root]
        p_mul = self.prompt(root_feat)[data.batch]
        p_add = self.prompt2(root_feat)[data.batch]

        # ---- prompts from tweet ----
        # p_mul = self.prompt(x)
        # p_add = self.prompt2(x)

        z = (1.0 - alpha) * (x * p_mul) + alpha * (x + p_add)

        TD_x, q1 = self.TDrumorGCN(z, data.edge_index, data.batch)
        BU_x, q2 = self.BUrumorGCN(z, data.edge_index, data.batch)
        h = torch.cat((BU_x, TD_x), 1)
        qs = torch.cat((q1, q2), 1)

        return h, qs

    def forward(self, *data_list):

        hs = []

        for data in data_list:
            h, _ = self.process_data(data)
            hs.append(h)

        h = torch.cat(hs, dim=0)
        h = self.proj_head(h)

        return h

    def loss_graphcl(self, x1, x2, mean=True):

        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def get_embeds(self, data):

        h, _ = self.process_data(data)

        return h

        h, gh = self.online_encoder(data, data.x)

        return gh

