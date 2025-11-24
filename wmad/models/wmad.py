"""MEAD: Multi-view Enhanced Anomaly Detection."""

from wmad.models.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import numpy as np
from collections import Counter
from deepod.utils.data import set_seed


class MEAD(BaseDeepAD):
    """
    Multi-view Enhanced Anomaly Detection (MEAD).
    
    A deep anomaly detection model for multi-view data with contrastive learning.
    
    Parameters
    ----------
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=64
        Batch size for training
    lr : float, default=1e-3
        Learning rate
    rep_dim : int, default=128
        Dimension of the representation space
    hidden_dims_view1 : str, default='100,50'
        Hidden layer dimensions for view 1
    hidden_dims_view2 : str, default='100,50'
        Hidden layer dimensions for view 2
    hidden_dims_view3 : str, default='100,50'
        Hidden layer dimensions for view 3
    act : str, default='ReLU'
        Activation function
    bias : bool, default=False
        Whether to use bias in network layers
    epoch_steps : int, default=-1
        Maximum steps per epoch (-1 for full epoch)
    prt_steps : int, default=10
        Print frequency (in epochs)
    device : str, default='cuda'
        Device to use ('cuda' or 'cpu')
    contamination : float, default=0.1
        Expected proportion of outliers
    verbose : int, default=2
        Verbosity level
    random_state : int, default=42
        Random seed
    model_name : str, default='MEAD'
        Model name
    eta1 : float, default=1
        Weight for normal samples
    eta2 : float, default=1
        Weight for anomalous samples
    lambda_ : float, default=1
        Weight for contrastive loss
    beta_ : float, default=1
        Weight for SVDD loss
    k : int, default=10
        Number of negative samples
    c : torch.Tensor or None, default=None
        Center of hypersphere (auto-initialized if None)
    use_inference_contrast : bool, default=False
        Whether to use cross-view contrast in inference
    inference_contrast_weight : float, default=0.5
        Weight for inference contrast
    temperature : float, default=1.0
        Temperature for contrastive learning
    contrastive_mode : str, default='cross_view_cross_sample'
        Mode for contrastive learning
    similarity_mode : str, default='distance'
        Similarity metric ('distance' or 'cosine')
    """

    def __init__(self, epochs=100, batch_size=64, lr=1e-3, rep_dim=128,
                 hidden_dims_view1='100,50', hidden_dims_view2='100,50', hidden_dims_view3='100,50',
                 act='ReLU', bias=False, epoch_steps=-1, prt_steps=10, device='cuda', contamination=0.1,
                 verbose=2, random_state=42, model_name='MEAD', eta1=1e-3, eta2=1e-3, lambda_=1e-2, beta_=1, k=10, c=None,
                 use_inference_contrast=False, inference_contrast_weight=0.5, temperature=1,
                 contrastive_mode='cross_view_cross_sample', similarity_mode='distance'):

        super(MEAD, self).__init__(
            data_type='tabular', model_name=model_name,
            epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, contamination=contamination
        )

        self.hidden_dims_view1 = hidden_dims_view1
        self.hidden_dims_view2 = hidden_dims_view2
        self.hidden_dims_view3 = hidden_dims_view3
        self.rep_dim = rep_dim
        self.X1_sel_neg = None
        self.X2_sel_neg = None
        self.X3_sel_neg = None
        self.act = act
        self.bias = bias
        self.eta1 = eta1
        self.eta2 = eta2
        self.lambda_ = lambda_
        self.beta_ = beta_
        self.k = k
        self.c = c
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.use_inference_contrast = use_inference_contrast
        self.inference_contrast_weight = inference_contrast_weight
        self.temperature = temperature
        self.contrastive_mode = contrastive_mode
        self.similarity_mode = similarity_mode

        set_seed(random_state)

        return

    def training_prepare(self, X1, X2, X3, y):

        # 标签处理
        has_unknown = False
        has_unknown = np.isnan(y).any()
        known_anom_id = np.where(y == 1)
        unknown_id = np.where(np.isnan(y))
        y = np.ones_like(y)
        y[known_anom_id] = -1
        y[unknown_id] = 0
        counter = Counter(y)

        # 数据准备
        sel_neg = known_anom_id[0]
        self.X1_sel_neg = torch.tensor(X1[sel_neg]).float()
        self.X2_sel_neg = torch.tensor(X2[sel_neg]).float()
        self.X3_sel_neg = torch.tensor(X3[sel_neg]).float()

        if self.verbose >= 2:
            print(f'training data counter: {counter}')
        dataset = TensorDataset(torch.from_numpy(X1).float(),
                                torch.from_numpy(X2).float(),
                                torch.from_numpy(X3).float(),
                                torch.from_numpy(y).long())
        if has_unknown:
            weight_map = {1: 1. / counter[1], -1: 1. / counter[-1], 0: 1./counter[0]}
        else:
            weight_map = {1: 1. / counter[1], -1: 1. / counter[-1]}
        sampler = WeightedRandomSampler(weights=[weight_map[label.item()] for data1, data2, data3, label in dataset],
                                        num_samples=len(dataset), replacement=True)
        train_loader = DataLoader(dataset, batch_size=self.batch_size,
                                  sampler=sampler,
                                  shuffle=True if sampler is None else False)

        # 网络准备
        network_params_view1 = {
            'n_features': self.n_features_view1,
            'n_hidden': self.hidden_dims_view1,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        network_params_view2 = {
            'n_features': self.n_features_view2,
            'n_hidden': self.hidden_dims_view2,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        network_params_view3 = {
            'n_features': self.n_features_view3,
            'n_hidden': self.hidden_dims_view3,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net1 = MLPnet(**network_params_view1).to(self.device)
        net2 = MLPnet(**network_params_view2).to(self.device)
        net3 = MLPnet(**network_params_view3).to(self.device)

        # 损失函数准备
        self.c, self.c1, self.c2, self.c3 = self._set_c(net1, net2, net3, train_loader)
        criterion = MEADLoss(c=self.c,c1=self.c1, c2=self.c2, c3 =self.c3,
                             eta1=self.eta1, eta2=self.eta2, lambda_=self.lambda_,beta_= self.beta_, k=self.k,
                             temperature=self.temperature, contrastive_mode=self.contrastive_mode,
                             similarity_mode=self.similarity_mode)

        # 打印所有超参数
        if self.verbose >= 0:
            print('=' * 80)
            print('MEAD 模型训练参数:')
            print('=' * 80)
            print(f'  训练参数:')
            print(f'    epochs: {self.epochs}')
            print(f'    batch_size: {self.batch_size}')
            print(f'    lr: {self.lr}')
            print(f'    rep_dim: {self.rep_dim}')
            print(f'  损失函数参数:')
            print(f'    eta1: {self.eta1}')
            print(f'    eta2: {self.eta2}')
            print(f'    lambda_: {self.lambda_}')
            print(f'    beta_: {self.beta_}')
            print(f'  对比学习参数:')
            print(f'    k: {self.k}')
            print(f'    temperature: {self.temperature}')
            print(f'    contrastive_mode: {self.contrastive_mode}')
            print(f'    similarity_mode: {self.similarity_mode}')
            print(f'  推理参数:')
            print(f'    use_inference_contrast: {self.use_inference_contrast}')
            print(f'    inference_contrast_weight: {self.inference_contrast_weight}')
            print(f'  网络结构:')
            print(f'    hidden_dims_view1: {self.hidden_dims_view1}')
            print(f'    hidden_dims_view2: {self.hidden_dims_view2}')
            print(f'    hidden_dims_view3: {self.hidden_dims_view3}')
            print('=' * 80)
        if self.verbose >= 2:
            print('\n网络结构详情:')
            print(net1)
            print(net2)
            print(net3)

        return train_loader, net1, net2, net3, criterion, self.c

    def inference_prepare(self, X1, X2, X3):

        dataset = TensorDataset(torch.from_numpy(X1).float(),
                                torch.from_numpy(X2).float(),
                                torch.from_numpy(X3).float())
        test_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'

        return test_loader

    def training_forward(self, batch_data, net1, net2, net3, criterion):
        batch_x1, batch_x2, batch_x3, batch_y = batch_data

        batch_x1 = batch_x1.float().to(self.device)
        batch_x2 = batch_x2.float().to(self.device)
        batch_x3 = batch_x3.float().to(self.device)
        batch_y = batch_y.long().to(self.device)

        z1 = net1(batch_x1)
        z2 = net2(batch_x2)
        z3 = net3(batch_x3)

        # 设置criterion返回详细损失字典
        criterion._return_dict = True
        loss_dict = criterion(z1, z2, z3, batch_y)
        criterion._return_dict = False  # 恢复默认行为

        # 返回总损失（用于反向传播）和详细损失字典
        return loss_dict['total'], loss_dict

    def inference_forward(self, batch_data, net1, net2, net3, criterion):
        batch_x1, batch_x2, batch_x3 = batch_data

        batch_x1 = batch_x1.float().to(self.device)
        batch_x2 = batch_x2.float().to(self.device)
        batch_x3 = batch_x3.float().to(self.device)

        z1 = net1(batch_x1)
        z2 = net2(batch_x2)
        z3 = net3(batch_x3)

        # 基础分数：到中心的距离
        s1 = criterion(z1, z2, z3)
        
        # 根据参数决定是否使用跨视角对比
        if self.use_inference_contrast:
            # 计算跨视角一致性分数（向量化实现，避免循环）
            eps = 1e-8
            # 提取对角线元素（同一样本的不同视角）
            sim_12_mat = criterion.compute_similarity(z1, z2)
            sim_12 = torch.diagonal(sim_12_mat)
            sim_13_mat = criterion.compute_similarity(z1, z3)
            sim_13 = torch.diagonal(sim_13_mat)
            sim_23_mat = criterion.compute_similarity(z2, z3)
            sim_23 = torch.diagonal(sim_23_mat)
            
            # 一致性分数：相似度低（距离大）则分数高
            s2 = -torch.log(sim_12 + eps) - torch.log(sim_13 + eps) - torch.log(sim_23 + eps)
            
            # 融合两部分分数
            s = s1 + self.inference_contrast_weight * s2
        else:
            s = s1
        
        if torch.isnan(s).any():
            raise ValueError("s contains NaN values")
        
        return z1, z2, z3, s

    def _set_c(self, net1, net2, net3, dataloader, eps=0.0):
        """Initializing the center for the hypersphere"""
        net1.eval()
        net2.eval()
        net3.eval()
        z_ = []
        z1_ = []
        z2_ = []
        z3_ = []
        with torch.no_grad():
            for x1, x2, x3, _ in dataloader:
                x1 = x1.float().to(self.device)
                x2 = x2.float().to(self.device)
                x3 = x3.float().to(self.device)
                z1 = net1(x1)
                z2 = net2(x2)
                z3 = net3(x3)
                z = (z1 + z2 + z3) / 3
                z_.append(z.detach())
                z1_.append(z1.detach())
                z2_.append(z2.detach())
                z3_.append(z3.detach())
        z_ = torch.cat(z_)
        z1_ = torch.cat(z1_)
        z2_ = torch.cat(z2_)
        z3_ = torch.cat(z3_)
        c = torch.mean(z_, dim=0)
        c1 = torch.mean(z1_, dim=0)
        c2 = torch.mean(z2_, dim=0)
        c3 = torch.mean(z3_, dim=0)
        

        # if c is too close to zero, set to +- eps
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        c1[(abs(c1) < eps) & (c1 < 0)] = -eps
        c1[(abs(c1) < eps) & (c1 > 0)] = eps
        c2[(abs(c2) < eps) & (c2 < 0)] = -eps
        c2[(abs(c2) < eps) & (c2 > 0)] = eps
        c3[(abs(c3) < eps) & (c3 < 0)] = -eps
        c3[(abs(c3) < eps) & (c3 > 0)] = eps
        
        return c, c1, c2, c3

    def save_model(self, path='./saved_models/mead.pth'):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'net1_state_dict': self.net1.state_dict(),
            'net2_state_dict': self.net2.state_dict(),
            'net3_state_dict': self.net3.state_dict(),
            'c': self.c,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
        }, path)

    def load_model(self, path='./saved_models/mead.pth'):
        checkpoint = torch.load(path)
        self.net1.load_state_dict(checkpoint['net1_state_dict'])
        self.net2.load_state_dict(checkpoint['net2_state_dict'])
        self.net3.load_state_dict(checkpoint['net3_state_dict'])
        self.c = checkpoint['c']
        self.c1 = checkpoint['c1']
        self.c2 = checkpoint['c2']
        self.c3 = checkpoint['c3']


class MEADLoss(torch.nn.Module):
    """
    Loss function for MEAD.
    
    Combines Deep SVDD loss with contrastive learning for multi-view anomaly detection.
    
    Parameters
    ----------
    c : torch.Tensor
        Center of the pre-defined hyper-sphere
    c1, c2, c3 : torch.Tensor
        Centers for each view
    eta1 : float
        Weight for normal samples
    eta2 : float
        Weight for anomalous samples
    eps : float, default=0
        Small epsilon for numerical stability
    lambda_ : float, default=1
        Weight for contrastive loss
    beta_ : float, default=1
        Weight for SVDD loss
    k : int, default=10
        Number of negative samples
    temperature : float, default=1.0
        Temperature for contrastive learning
    contrastive_mode : str, default='cross_view_cross_sample'
        Mode for contrastive learning
    similarity_mode : str, default='distance'
        Similarity metric
    reduction : str, default='mean'
        Loss reduction mode
    """

    def __init__(self, c, c1, c2, c3, eta1, eta2, eps=0, lambda_=1, beta_ = 1, k=10, temperature=1.0, 
                 contrastive_mode='cross_view_cross_sample', similarity_mode='distance', reduction='mean'):
        super(MEADLoss, self).__init__()
        self.c = c
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.reduction = reduction
        self.eta1 = eta1
        self.eta2 = eta2
        self.eps = eps
        self.lambda_ = lambda_
        self.k = k
        self.beta = beta_
        self.temperature = temperature
        self.contrastive_mode = contrastive_mode
        self.similarity_mode = similarity_mode
    
    def compute_similarity(self, rep1, rep2):
        """
        计算两个表示之间的相似度
        
        Parameters
        ----------
        rep1: torch.Tensor
            第一个表示 [n1, dim]
        rep2: torch.Tensor
            第二个表示 [n2, dim] 或 [n1, dim]
        
        Returns
        -------
        similarity: torch.Tensor
            相似度值
        """
        # 确保输入是2维的
        if rep1.dim() == 1:
            rep1 = rep1.unsqueeze(0)
        if rep2.dim() == 1:
            rep2 = rep2.unsqueeze(0)
        
        if self.similarity_mode == 'cosine':
            # 余弦相似度
            rep1_norm = rep1 / (torch.norm(rep1, dim=-1, keepdim=True) + self.eps)
            rep2_norm = rep2 / (torch.norm(rep2, dim=-1, keepdim=True) + self.eps)
            similarity = torch.mm(rep1_norm, rep2_norm.t())
            return torch.exp(similarity / self.temperature)
        else:  # 'distance'
            # 基于距离的相似度
            dist = torch.cdist(rep1, rep2)
            return torch.exp(-dist / self.temperature)

    def forward(self, rep1, rep2, rep3, semi_targets=None, reduction=None):

        dist1 = torch.sum((rep1 - self.c) ** 2, dim=1)
        dist2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        dist3 = torch.sum((rep3 - self.c) ** 2, dim=1)

        if semi_targets is not None:

            # dsad loss
            if self.beta != 0:
                loss1 = dist1
                loss1 = torch.where(semi_targets == 1,  self.eta1 * ((dist1+self.eps) ** semi_targets.float()), loss1)
                loss1 = torch.where(semi_targets == -1, self.eta2 * ((dist1+self.eps) ** semi_targets.float()), loss1)
                loss1 = torch.mean(loss1)
                loss2 = dist2
                loss2 = torch.where(semi_targets == 1,  self.eta1 * ((dist2+self.eps) ** semi_targets.float()), loss2)
                loss2 = torch.where(semi_targets == -1, self.eta2 * ((dist2+self.eps) ** semi_targets.float()), loss2)
                loss2 = torch.mean(loss2)
                loss3 = dist3
                loss3 = torch.where(semi_targets == 1,  self.eta1 * ((dist3+self.eps) ** semi_targets.float()), loss3)
                loss3 = torch.where(semi_targets == -1, self.eta2 * ((dist3+self.eps) ** semi_targets.float()), loss3)
                loss3 = torch.mean(loss3)
            else:
                loss1 = 0.0
                loss2 = 0.0
                loss3 = 0.0

            # contrastive loss - 向量化实现
            if self.lambda_ != 0:
                pos = torch.where(semi_targets == 1)[0]
                neg = torch.where(semi_targets == -1)[0]
                
                # 限制负样本数量
                if len(neg) > self.k:
                    neg = neg[:self.k]
                
                # 需要有正样本和负样本才能计算对比损失
                if len(pos) > 0 and len(neg) > 0:
                    # 提取正负样本的表示
                    rep1_pos = rep1[pos]
                    rep2_pos = rep2[pos]
                    rep3_pos = rep3[pos]
                    
                    rep1_neg = rep1[neg]
                    rep2_neg = rep2[neg]
                    rep3_neg = rep3[neg]
                    
                    # 正对：正常样本的不同视角应该相似
                    sim_pos_12_mat = self.compute_similarity(rep1_pos, rep2_pos)
                    sim_pos_12 = torch.diagonal(sim_pos_12_mat)
                    sim_pos_13_mat = self.compute_similarity(rep1_pos, rep3_pos)
                    sim_pos_13 = torch.diagonal(sim_pos_13_mat)
                    sim_pos_23_mat = self.compute_similarity(rep2_pos, rep3_pos)
                    sim_pos_23 = torch.diagonal(sim_pos_23_mat)
                    
                    eps = 1e-8
                    
                    if self.contrastive_mode == 'same_view_cross_sample':
                        # 同视角跨样本对比 (InfoNCE style)
                        sim_neg_1 = self.compute_similarity(rep1_pos, rep1_neg).sum(dim=1)
                        sim_neg_2 = self.compute_similarity(rep2_pos, rep2_neg).sum(dim=1)
                        sim_neg_3 = self.compute_similarity(rep3_pos, rep3_neg).sum(dim=1)
                        
                        neg_12 = (sim_neg_1 + sim_neg_2) / 2
                        neg_13 = (sim_neg_1 + sim_neg_3) / 2
                        neg_23 = (sim_neg_2 + sim_neg_3) / 2
                        
                        # InfoNCE: -log(pos / (pos + neg))
                        contrast_loss = -torch.log(sim_pos_12 / (sim_pos_12 + neg_12 + eps)) \
                                       -torch.log(sim_pos_13 / (sim_pos_13 + neg_13 + eps)) \
                                       -torch.log(sim_pos_23 / (sim_pos_23 + neg_23 + eps))
                        
                    else:  # 'cross_view_cross_sample' (默认)
                        # 跨视角跨样本对比 (InfoNCE style)
                        sim_neg_12 = self.compute_similarity(rep1_pos, rep2_neg).sum(dim=1)
                        sim_neg_13 = self.compute_similarity(rep1_pos, rep3_neg).sum(dim=1)
                        sim_neg_23 = self.compute_similarity(rep2_pos, rep3_neg).sum(dim=1)
                        sim_neg_21 = self.compute_similarity(rep2_pos, rep1_neg).sum(dim=1)
                        sim_neg_31 = self.compute_similarity(rep3_pos, rep1_neg).sum(dim=1)
                        sim_neg_32 = self.compute_similarity(rep3_pos, rep2_neg).sum(dim=1)
                        
                        # InfoNCE: -log(pos / (pos + neg))，确保损失非负
                        contrast_loss = -torch.log(sim_pos_12 / (sim_pos_12 + sim_neg_12 + eps)) \
                                       -torch.log(sim_pos_13 / (sim_pos_13 + sim_neg_13 + eps)) \
                                       -torch.log(sim_pos_23 / (sim_pos_23 + sim_neg_23 + eps)) \
                                       -torch.log(sim_pos_12 / (sim_pos_12 + sim_neg_21 + eps)) \
                                       -torch.log(sim_pos_13 / (sim_pos_13 + sim_neg_31 + eps)) \
                                       -torch.log(sim_pos_23 / (sim_pos_23 + sim_neg_32 + eps))
                    
                    contrast_loss = contrast_loss.mean()
                else:
                    contrast_loss = torch.tensor(0.0, device=rep1.device)
            else:
                contrast_loss = torch.tensor(0.0, device=rep1.device)

            # total loss
            svdd_loss = loss1 + loss2 + loss3
            constant_term = torch.tensor(1e-6, dtype=torch.float32, requires_grad=True, device=rep1.device)
            total_loss = self.beta * svdd_loss + self.lambda_ * contrast_loss + constant_term
            
            loss_dict = {
                'total': total_loss,
                'svdd': svdd_loss,
                'svdd_view1': loss1,
                'svdd_view2': loss2,
                'svdd_view3': loss3,
                'contrast': contrast_loss,
                'beta': self.beta,
                'lambda': self.lambda_,
            }
            
        else:
            total_loss = dist1 + dist2 + dist3
            loss_dict = {
                'total': total_loss,
                'svdd': total_loss,
                'svdd_view1': dist1.mean(),
                'svdd_view2': dist2.mean(),
                'svdd_view3': dist3.mean(),
                'contrast': torch.tensor(0.0, device=rep1.device),
                'beta': 1.0,
                'lambda': 0.0,
            }

        if reduction is None:
            reduction = self.reduction
        
        # 如果需要返回详细损失，返回字典
        if hasattr(self, '_return_dict') and self._return_dict:
            if reduction == 'mean':
                return {k: torch.mean(v) if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            elif reduction == 'sum':
                return {k: torch.sum(v) if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            else:
                return loss_dict
        
        # 否则只返回总损失
        if reduction == 'mean':
            return torch.mean(loss_dict['total'])
        elif reduction == 'sum':
            return torch.sum(loss_dict['total'])
        elif reduction == 'none':
            return loss_dict['total']

