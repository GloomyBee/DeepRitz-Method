"""
多层感知机实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from ..data_utils.sampler import Utils


class EnhancedRitzNet(BaseModel):
    """增强的Ritz网络，包含高斯核特征"""
    
    def __init__(self, params: dict, nodes: torch.Tensor, s: torch.Tensor):
        super().__init__(params)
        self.device = params["device"]
        self.nodes = nodes.clone().detach().to(self.device)
        self.s = s.clone().detach().to(self.device)
        self.num_nodes = len(nodes)

        # 网络结构
        self.linear_in = nn.Linear(params["d"] + self.num_nodes, params["width"])
        self.linear = nn.ModuleList([nn.Linear(params["width"], params["width"]) for _ in range(params["depth"])])
        self.linear_out = nn.Linear(params["width"], params["dd"])

    def compute_kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算高斯核特征
        
        Args:
            x: 输入张量 [batch_size, 2]
            
        Returns:
            核特征张量 [batch_size, num_nodes]
        """
        dx = x[:, 0:1] - self.nodes[:, 0:1].T
        dy = x[:, 1:2] - self.nodes[:, 1:2].T
        r = torch.sqrt(dx**2 + dy**2)
        features = Utils.gaussian_kernel(r, self.s, self.device)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, 2]
            
        Returns:
            输出张量 [batch_size, 1]
        """
        kernel_features = self.compute_kernel_features(x)
        x_enhanced = torch.cat((x, kernel_features), dim=1)
        x = torch.tanh(self.linear_in(x_enhanced))
        for layer in self.linear:
            x = torch.tanh(layer(x))
        return self.linear_out(x)