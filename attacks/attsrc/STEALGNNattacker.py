"""STEALGNN Type II：硬标签监督、余弦建图与单代理梯度。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.models import GAT, GCN, GraphSAGE


@dataclass(frozen=True)
class STEALGNNConfig:
    num_nodes: int = 250
    noise_dim: int = 64
    threshold: float = 0.1
    generator_steps: int = 2
    surrogate_steps: int = 5
    generator_lr: float = 1e-6
    surrogate_lr: float = 0.001

    def __post_init__(self):
        for name in (
            "num_nodes", "noise_dim", "generator_steps", "surrogate_steps",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int) or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"STEALGNN {name} 必须为正整数")
        if not 0 <= self.threshold <= 1:
            raise ValueError("STEALGNN 余弦阈值必须在 [0, 1] 内")
        for name in ("generator_lr", "surrogate_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"STEALGNN {name} 必须为有限正数")


class STEALGNNGenerator(nn.Module):
    """将整图高斯噪声映射为连续特征，并按余弦阈值构造无向边。"""

    def __init__(self, feature_dim, config):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config
        self.feature_generator = nn.Sequential(
            nn.Linear(config.noise_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, config.num_nodes * feature_dim), nn.Tanh(),
        )

    def forward(self):
        parameter = next(self.parameters())
        noise = torch.randn(
            1, self.config.noise_dim, device=parameter.device,
            dtype=parameter.dtype,
        )
        features = self.feature_generator(noise).reshape(
            self.config.num_nodes, self.feature_dim,
        )
        # 离散边每次按新特征重建；生成器梯度经代理的特征输入传播。
        with torch.no_grad():
            normalized = F.normalize(features, dim=1)
            adjacency = normalized @ normalized.T > self.config.threshold
            adjacency.fill_diagonal_(False)
            edges = adjacency.nonzero().T.contiguous()
        return features, edges


class STEALGNNModel(nn.Module):
    """随机初始化的两层代理 GNN，输出节点对数概率。"""

    def __init__(self, input_dim, hidden_dim, output_dim, model_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        architecture = {"GCN": GCN, "GAT": GAT, "SAGE": GraphSAGE}[model_type]
        self.encoder = architecture(
            in_channels=input_dim, hidden_channels=hidden_dim,
            num_layers=2, out_channels=output_dim, dropout=0.5,
        )

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index).log_softmax(dim=-1)


def query_hard_labels(target, features, edges, num_classes):
    """仅返回硬标签；查询期间关闭梯度及训练态，随后恢复原有模式。"""
    modes = [(module, module.training) for module in target.modules()]
    try:
        target.eval()
        with torch.no_grad():
            output = target(features, edges)
            if output.shape != (features.shape[0], num_classes):
                raise ValueError("STEALGNN 目标响应与节点数或类别数不匹配")
            if not torch.isfinite(output).all():
                raise ValueError("STEALGNN 目标响应包含非有限值")
            return output.argmax(dim=1)
    finally:
        for module, mode in modes:
            module.training = mode


def train_generator_step(generator, surrogate, target, optimizer):
    """固定代理参数，通过其输入梯度增大与 Target 硬标签的交叉熵。"""
    generator.train()
    surrogate.eval()
    optimizer.zero_grad(set_to_none=True)
    features, edges = generator()
    labels = query_hard_labels(target, features, edges, surrogate.output_dim)
    loss = -F.nll_loss(surrogate(features, edges), labels)
    if not torch.isfinite(loss):
        raise ValueError("STEALGNN 生成器损失包含非有限值")
    # 只累积生成器参数梯度，保留经过代理网络的完整链式求导。
    torch.autograd.backward(loss, inputs=tuple(generator.parameters()))
    optimizer.step()
    return loss.item()


def train_surrogate_step(generator, surrogate, target, optimizer):
    """固定生成器，以合成图及 Target 硬标签训练代理的全部层。"""
    generator.eval()
    surrogate.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        features, edges = generator()
    labels = query_hard_labels(target, features, edges, surrogate.output_dim)
    loss = F.nll_loss(surrogate(features, edges), labels)
    if not torch.isfinite(loss):
        raise ValueError("STEALGNN 代理损失包含非有限值")
    loss.backward()
    optimizer.step()
    return loss.item()


def train_stealgnn(surrogate, target, generator, *, epochs):
    """按固定轮数交替优化；训练接口仅接收模型和合成图生成器。"""
    if epochs < 1:
        raise ValueError("STEALGNN 训练轮数必须为正")
    config = generator.config
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.generator_lr,
    )
    surrogate_optimizer = torch.optim.Adam(
        surrogate.parameters(), lr=config.surrogate_lr,
    )
    history = []
    for epoch in range(epochs):
        generator_loss = sum(
            train_generator_step(
                generator, surrogate, target, generator_optimizer,
            )
            for _ in range(config.generator_steps)
        ) / config.generator_steps
        surrogate_loss = sum(
            train_surrogate_step(
                generator, surrogate, target, surrogate_optimizer,
            )
            for _ in range(config.surrogate_steps)
        ) / config.surrogate_steps
        history.append({
            "epoch": epoch + 1, "generator_loss": generator_loss,
            "surrogate_loss": surrogate_loss,
        })
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"STEALGNN 轮次 {epoch + 1}/{epochs}，"
                f"生成器损失={generator_loss:.6f}，"
                f"代理损失={surrogate_loss:.6f}"
            )
    surrogate.eval()
    generator.eval()
    calls = epochs * (config.generator_steps + config.surrogate_steps)
    return {
        "extraction_target_calls": calls,
        "extraction_node_responses": calls * config.num_nodes,
        "generator_target_calls": epochs * config.generator_steps,
        "surrogate_target_calls": epochs * config.surrogate_steps,
        "history": history,
    }
