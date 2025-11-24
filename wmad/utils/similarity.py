"""Similarity computation functions for WMAD."""

import torch


def cos_sim(vec1, vec2):
    """
    计算余弦相似度（带温度缩放）。
    
    Parameters
    ----------
    vec1 : torch.Tensor
        一维张量
    vec2 : torch.Tensor
        一维或二维张量
        
    Returns
    -------
    torch.Tensor
        相似度值（经过exp变换）
    """
    t = 1
    # 确保 vec1 是一维张量
    if vec1.dim() != 1:
        raise ValueError("vec1 应该是一维张量")
    
    # 如果 vec2 是一维张量
    if vec2.dim() == 1:
        # 计算点积
        dot_product = torch.dot(vec1, vec2)
        # 计算向量的范数
        norm_vec1 = torch.norm(vec1)
        norm_vec2 = torch.norm(vec2)
        # 计算余弦相似度并返回
        return torch.exp((dot_product / (norm_vec1 * norm_vec2 + 1e-6)) / t)
    
    # 如果 vec2 是二维张量
    elif vec2.dim() == 2:
        # 计算点积
        dot_product = torch.matmul(vec2, vec1)
        # 计算向量的范数
        norm_vec1 = torch.norm(vec1)
        norm_vec2 = torch.norm(vec2, dim=1)
        # 计算余弦相似度并返回总和
        return torch.exp((dot_product / (norm_vec1 * norm_vec2 + 1e-6)) / t).sum()
    
    else:
        raise ValueError("vec2 应该是一维或二维张量")


def dis_sim(vec1, vec2):
    """
    计算基于欧式距离的相似度（带温度缩放）。
    
    Parameters
    ----------
    vec1 : torch.Tensor
        一维张量
    vec2 : torch.Tensor
        一维或二维张量
        
    Returns
    -------
    torch.Tensor
        相似度值（距离的负指数）
    """
    t = 1
    # 确保 vec1 是一维张量
    if vec1.dim() != 1:
        raise ValueError("vec1 应该是一维张量")

    # 如果 vec2 是一维张量
    if vec2.dim() == 1:
        # 计算欧式距离
        dist = torch.norm(vec1 - vec2)
        # 返回相似度（距离的负指数）
        return torch.exp(-dist / t)

    # 如果 vec2 是二维张量
    elif vec2.dim() == 2:
        # 计算欧式距离
        dist = torch.norm(vec2 - vec1, dim=1)
        # 返回相似度（距离的负指数）并返回总和
        return torch.exp(-dist / t).sum()

    else:
        raise ValueError("vec2 应该是一维或二维张量")


