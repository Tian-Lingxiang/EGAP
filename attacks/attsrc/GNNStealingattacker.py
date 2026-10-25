"""GNNStealing Type I.2 攻击：后验响应回归与独立标签分类器。"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.utils import subgraph

from src.detection.graph_io import normalize_log_probabilities


class GNNStealingModel(nn.Module):
    """将响应回归器和分类器保存为支持输入梯度的节点模型。"""

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
        self.classifier = nn.Identity()
        self.register_buffer("classes", torch.empty(0, dtype=torch.long))

    def forward(self, x, edge_index):
        if self.classes.numel() == 0:
            raise RuntimeError("GNNStealing 分类器尚未训练")
        logits = self.classifier(self.encoder(x, edge_index))
        if logits.shape[1] == 1:
            probabilities = torch.cat(
                ((-logits).sigmoid(), logits.sigmoid()), dim=1,
            )
        else:
            probabilities = logits.softmax(dim=1)
        # sklearn 的列序由 classes_ 定义，必须映射回沙盒的全局类别编号。
        output = probabilities.new_zeros((x.shape[0], self.output_dim))
        output = output.index_copy(1, self.classes, probabilities)
        output = output.clamp_min(1e-8)
        return (output / output.sum(dim=1, keepdim=True)).log()

    def set_classifier(self, classifier):
        """精确转换 sklearn 的 ReLU MLP，保留检测阶段的输入梯度。"""
        device = next(self.encoder.parameters()).device
        dtype = next(self.encoder.parameters()).dtype
        layers = []
        for index, (weight, bias) in enumerate(
            zip(classifier.coefs_, classifier.intercepts_)
        ):
            linear = nn.Linear(*weight.shape, device=device, dtype=dtype)
            with torch.no_grad():
                linear.weight.copy_(torch.as_tensor(weight.T, device=device))
                linear.bias.copy_(torch.as_tensor(bias, device=device))
            layers.append(linear)
            if index < len(classifier.coefs_) - 1:
                layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*layers)
        self.classes = torch.as_tensor(
            classifier.classes_, device=device, dtype=torch.long,
        )


def target_response(target, data):
    """以评估模式查询节点后验，并恢复 Target 的原有模式。"""
    modes = [(module, module.training) for module in target.modules()]
    try:
        target.eval()
        with torch.no_grad():
            return normalize_log_probabilities(
                target(data.x, data.edge_index)
            ).exp().detach()
    finally:
        for module, mode in modes:
            module.training = mode


def query_subgraph(data, query_ratio, seed):
    """从训练分区抽取约 0.3N × ratio 个节点，构造节点诱导子图。"""
    if not 0 < query_ratio <= 1:
        raise ValueError("GNNStealing query_ratio 必须在 (0, 1] 内")
    masks = [data.train_mask, data.val_mask, data.test_mask]
    if any(
        mask.dtype != torch.bool or mask.shape != (data.num_nodes,)
        for mask in masks
    ):
        raise ValueError("GNNStealing 需要一维布尔训练、验证、测试掩码")
    if (sum(mask.int() for mask in masks) > 1).any():
        raise ValueError("GNNStealing 的训练、验证、测试分区必须互斥")
    count = int(int(data.num_nodes * 0.3) * query_ratio)
    candidates = data.train_mask.nonzero(as_tuple=True)[0].cpu()
    if count < 2 or count > candidates.numel():
        raise ValueError("GNNStealing 查询预算不足或超过训练候选节点数")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(candidates.numel(), generator=generator)
    nodes = candidates[permutation[:count]]
    nodes = nodes.to(data.x.device)
    edges, _ = subgraph(
        nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes,
    )
    query = Data(
        x=data.x[nodes], edge_index=edges, y=data.y.reshape(-1)[nodes],
    )
    return query, nodes


def train_gnnstealing(model, target, query, *, epochs, seed):
    """缓存一次节点后验，回归表示，再用真实标签拟合独立 MLP。"""
    if epochs < 1:
        raise ValueError("GNNStealing 训练轮数必须为正")
    labels = query.y.detach().cpu().numpy()
    if np.unique(labels).size < 2:
        raise ValueError("GNNStealing 查询节点必须覆盖至少两个类别")
    if labels.min() < 0 or labels.max() >= model.output_dim:
        raise ValueError("查询节点标签超出 Target 的类别范围")
    # 对齐参考实现：分类器使用查询图的 75% 节点，25% 用于内部验证。
    train_ids, validation_ids = train_test_split(
        np.arange(query.num_nodes), stratify=labels, random_state=seed,
    )
    responses = target_response(target, query)
    if responses.shape != (query.num_nodes, model.output_dim):
        raise ValueError("目标响应与 GNNStealing 节点数或类别数不匹配")

    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model.encoder(query.x, query.edge_index)
        # norm 的零点次梯度有限，避免 sqrt(MSE) 在精确匹配时产生 NaN。
        loss = (
            torch.linalg.vector_norm(prediction - responses)
            / responses.numel() ** 0.5
        )
        if not torch.isfinite(loss):
            raise ValueError("GNNStealing 回归损失包含非有限值")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"GNNStealing 轮次 {epoch + 1}/{epochs}，"
                f"响应 RMSE={loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        representations = model.encoder(query.x, query.edge_index)
        representations = representations.cpu().numpy()
    classifier = MLPClassifier(random_state=seed, max_iter=300)
    classifier.fit(representations[train_ids], labels[train_ids])
    validation_accuracy = classifier.score(
        representations[validation_ids], labels[validation_ids],
    )
    model.set_classifier(classifier)
    model.eval()
    return {
        "response_rmse": float(loss.detach()),
        "classifier_validation_accuracy": float(validation_accuracy),
        "classifier_iterations": int(classifier.n_iter_),
        "classifier_loss": float(classifier.loss_),
        "classifier_train_local_nodes": train_ids.tolist(),
        "classifier_validation_local_nodes": validation_ids.tolist(),
        "extraction_target_calls": 1,
        "extraction_node_responses": query.num_nodes,
    }
