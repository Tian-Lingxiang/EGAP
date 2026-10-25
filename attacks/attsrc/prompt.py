import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPrompt(nn.Module):
    def __init__(
        self,
        prompt_nodes_num,
        feature_dim,
        inner_prune,
        cross_prune,
        device,
        reference_features=None,
        cross_top_k=10,
    ):
        super(GraphPrompt, self).__init__()
        self.device = device
        self.prompt_nodes_num = prompt_nodes_num
        self.feature_dim = feature_dim
        self.inner_prune = inner_prune
        self.cross_prune = cross_prune
        self.cross_top_k = cross_top_k
        self.constrain_features = reference_features is not None

        if reference_features is None:
            prompt_features = torch.empty(prompt_nodes_num, feature_dim, device=device)
            torch.nn.init.kaiming_uniform_(
                prompt_features, nonlinearity="leaky_relu", mode="fan_in", a=0.01
            )
            support_mask = torch.ones_like(prompt_features)
            nonnegative = False
            target_norm = torch.tensor(1.0, device=device)
        else:
            if reference_features.ndim != 2 or reference_features.size(1) != feature_dim:
                raise ValueError("reference_features must have shape [K, feature_dim]")
            indices = torch.randint(
                reference_features.size(0),
                (prompt_nodes_num,),
                device=reference_features.device,
            )
            prompt_features = reference_features[indices].detach().clone()
            support_mask = (prompt_features != 0).to(prompt_features.dtype)
            nonnegative = bool(reference_features.min().item() >= 0)
            target_norm = reference_features.norm(dim=1).median().clamp_min(1e-12)

        self.prompt_features = nn.Parameter(prompt_features)
        self.register_buffer("support_mask", support_mask)
        self.register_buffer("target_norm", target_norm.detach().clone())
        self.nonnegative = nonnegative

    @torch.no_grad()
    def project_features(self):
        """Keep learned prompt features in the raw feature domain."""
        if not self.constrain_features:
            return
        if self.nonnegative:
            self.prompt_features.clamp_(min=0)
        self.prompt_features.mul_(self.support_mask)
        row_norms = self.prompt_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.prompt_features.mul_(self.target_norm / row_norms)

    def build_inner_structure(self):
        with torch.no_grad():
            normed_feat = F.normalize(self.prompt_features, p=2, dim=1)
            sim_matrix = torch.matmul(normed_feat, normed_feat.T)
            sim_matrix.fill_diagonal_(-float("inf"))
            inner_adj = torch.where(sim_matrix < self.inner_prune, 0, sim_matrix)
            inner_edge_index = inner_adj.nonzero().t().contiguous()

        return inner_adj, inner_edge_index

    def get_prompted_graph(self, data_feature, data_edge_index):
        inner_adj, inner_edge_index = self.build_inner_structure()
        prompt_normed_feat = F.normalize(self.prompt_features, p=2, dim=1)
        data_normed_feat = F.normalize(data_feature, p=2, dim=1)
        cross_dot = torch.matmul(prompt_normed_feat, data_normed_feat.T)
        if self.cross_top_k is None:
            # Match the original GrAP3 prompt: retain every cross edge whose
            # sigmoid-transformed cosine similarity exceeds cross_prune.
            cross_sim = torch.sigmoid(cross_dot)
            cross_edge_index = (
                (cross_sim >= self.cross_prune).nonzero().t().contiguous()
            )
        else:
            neighbor_count = min(self.cross_top_k, data_feature.size(0))
            cross_neighbors = torch.topk(
                cross_dot, k=neighbor_count, dim=1
            ).indices
            prompt_indices = torch.arange(
                self.prompt_nodes_num, device=data_feature.device
            ).unsqueeze(1).expand_as(cross_neighbors)
            cross_edge_index = torch.stack(
                [prompt_indices.reshape(-1), cross_neighbors.reshape(-1)], dim=0
            )

        # 统一索引编号
        cross_edge_index[0] += data_feature.shape[0]
        cross_edge_index[1] = cross_edge_index[1]  # 训练图中的节点索引不变
        reversed_edge_index = cross_edge_index[[1, 0], :]
        full_edge_index = torch.cat([cross_edge_index, reversed_edge_index], dim=1)

        inner_edge_index = inner_edge_index + data_feature.shape[0]

        new_edge_index = torch.cat(
            [data_edge_index, full_edge_index, inner_edge_index], dim=1
        )
        new_feature = torch.cat([data_feature, self.prompt_features], dim=0)

        return new_feature, new_edge_index

    def forward(self, data_feature, data_edge_index):
        new_feature, new_edge_index = self.get_prompted_graph(
            data_feature, data_edge_index
        )
        return new_feature, new_edge_index


if __name__ == "__main__":
    num_data_nodes = 5
    prompt_nodes_num = 3
    feature_dim = 4
    inner_prune = 0.3
    cross_prune = 0.4
    device = torch.device("cpu")

    # 创建 data_feature 和 data_edge_index
    data_feature = F.normalize(torch.rand(num_data_nodes, feature_dim), dim=1)
    data_edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )  # 假设存在4条边

    # 初始化 GraphPrompt 模块
    model = GraphPrompt(
        prompt_nodes_num=prompt_nodes_num,
        feature_dim=feature_dim,
        inner_prune=inner_prune,
        cross_prune=cross_prune,
        device=device,
    )

    # 调用 forward 函数
    new_feature, new_edge_index = model(data_feature, data_edge_index)

    # 输出信息检查
    print("原始节点数:", num_data_nodes)
    print("提示节点数:", prompt_nodes_num)
    print("输出特征矩阵 shape:", new_feature.shape)
    print("输出边索引 shape:", new_edge_index.shape)
    print("边索引:\n", new_edge_index)
