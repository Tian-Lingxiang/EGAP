import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGenerator(nn.Module):
    def __init__(
        self,
        num_nodes,
        feature_dim,
        class_num,
        device,
        top_k=10,
        dataset_prompt=None,
        feature_mode="grap3",
    ):
        super(GraphGenerator, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.sigma = 1.0
        self.center_dist = 3.5
        self.top_k = top_k
        self.intra_ratio = 0.7
        self.inter_ratio = 0.3
        self.node_labels = torch.randint(0, class_num, (num_nodes,))
        self.dataset_prompt = dict(dataset_prompt or {})
        if feature_mode not in {"grap3", "schema"}:
            raise ValueError("feature_mode must be either 'grap3' or 'schema'")
        self.feature_mode = feature_mode
        prompt_text = " ".join(
            [
                str(self.dataset_prompt.get("description", "")),
                str(self.dataset_prompt.get("feature_type", "")),
            ]
        ).lower()
        self.is_sparse = any(
            token in prompt_text for token in ("sparse", "bag-of-words", "tfidf")
        )
        self.is_nonnegative = any(
            token in prompt_text
            for token in ("non-negative", "nonnegative", "binary", "bag-of-words", "tfidf")
        )
        self.is_binary = "binary" in prompt_text
        # A domain-independent sparse prior: no target samples or private
        # statistics are consulted. It yields at most 64 active dimensions and
        # caps density at 2% for high-dimensional raw text attributes.
        self.feature_density = min(0.02, 64.0 / max(1, self.feature_dim))
        self.target_norm = 1.0
        self.directed = bool(self.dataset_prompt.get("directed", False))
        self.node_features = nn.Parameter(self.init_features(), requires_grad=True)
        # self.feature_gen = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_nodes * feature_dim),
        #     nn.Tanh()
        # )

    def init_features(self):
        if self.feature_mode == "schema":
            return self.sample_synthetic_features(self.num_nodes)

        means = []
        for _ in range(self.class_num):
            vec = np.random.randn(self.feature_dim)
            vec = vec / np.linalg.norm(vec)
            means.append(vec)
        means = self.center_dist * self.sigma * np.stack(means)
        class_means = torch.tensor(means, dtype=torch.float32)
        features = []
        for i in range(self.num_nodes):
            mean = class_means[self.node_labels[i]]
            noise = torch.randn(self.feature_dim) * self.sigma
            features.append(mean + noise)
        return torch.stack(features)

    def sample_grap3_features(self, node_labels):
        """Sample the multi-class CSBM features described in the paper."""
        class_means = torch.randn(
            self.class_num,
            self.feature_dim,
            device=self.device,
        )
        class_means = F.normalize(class_means, p=2, dim=1)
        class_means = class_means * self.center_dist * self.sigma
        noise = torch.randn(
            node_labels.numel(),
            self.feature_dim,
            device=self.device,
        ) * self.sigma
        return class_means[node_labels] + noise

    def sample_synthetic_features(self, sample_count):
        """Synthesize features using only public text/schema information."""
        device = self.device
        dtype = torch.float32

        if self.is_sparse:
            mask = torch.rand(sample_count, self.feature_dim, device=device)
            mask = mask < self.feature_density
            empty_rows = ~mask.any(dim=1)
            if empty_rows.any():
                fallback_columns = torch.randint(
                    self.feature_dim, (int(empty_rows.sum()),), device=device
                )
                mask[empty_rows, fallback_columns] = True

            features = torch.zeros(
                sample_count,
                self.feature_dim,
                dtype=dtype,
                device=device,
            )
            if self.is_binary:
                features[mask] = 1.0
            else:
                features[mask] = torch.rand(int(mask.sum()), device=device)
        else:
            features = torch.randn(
                sample_count,
                self.feature_dim,
                dtype=dtype,
                device=device,
            )
            if self.is_nonnegative:
                features.clamp_(min=0)

        row_norms = features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return features * (self.target_norm / row_norms)

    def get_adjacency_by_similarity(self, features=None, node_labels=None):
        with torch.no_grad():
            if features is None:
                features = self.node_features
            if node_labels is None:
                node_labels = self.node_labels.to(features.device)
            normed_feat = F.normalize(features, p=2, dim=1)
            sim_matrix = torch.matmul(normed_feat, normed_feat.T)
            adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32).to(
                self.device
            )
            for i in range(self.num_nodes):
                label_i = node_labels[i]
                sim_row = sim_matrix[i]

                same_class_mask = node_labels == label_i
                same_class_mask[i] = False
                diff_class_mask = ~same_class_mask
                diff_class_mask[i] = False

                n_intra = min(
                    int(self.top_k * self.intra_ratio), same_class_mask.sum().item()
                )
                n_inter = min((self.top_k - n_intra), diff_class_mask.sum().item())

                # 获取 top-k 类内 / 类间节点索引
                intra_idx = torch.topk(
                    sim_row.masked_fill(diff_class_mask, -1e9), n_intra
                ).indices
                inter_idx = torch.topk(
                    sim_row.masked_fill(same_class_mask, -1e9), n_inter
                ).indices

                neighbors = torch.cat([intra_idx, inter_idx])
                adj[i, neighbors] = 1.0

            if not self.directed:
                adj = ((adj + adj.T) > 0).float()
            edge_index = adj.nonzero(as_tuple=False).T  # shape: [2, num_edges]
            return sim_matrix, adj, edge_index

    def get_adj_wo_labels(self,features):
        with torch.no_grad():
            normed_feat = F.normalize(features, p=2, dim=1)
            sim_matrix = torch.matmul(normed_feat, normed_feat.T)
            adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32).to(
                self.device
            )
            # 取top-k相似度最大的节点作为邻居
            for i in range(self.num_nodes):
                sim_row = sim_matrix[i]
                sim_row = sim_row.clone()
                sim_row[i] = -float("inf")
                neighbor_count = min(self.top_k, self.num_nodes - 1)
                neighbors = torch.topk(sim_row, neighbor_count).indices
                adj[i, neighbors] = 1.0
            if not self.directed:
                adj = ((adj + adj.T) > 0).float()
            edge_index = adj.nonzero(as_tuple=False).T  # shape: [2, num_edges]
            return sim_matrix, adj, edge_index
        
    # def show_graph_heatmap(self, sim_matrix, adj, tag):
    #     labels_np = self.node_labels.cpu().numpy()
    #     sorted_idx = np.argsort(labels_np)
    #     adj_np = adj.cpu().numpy()
    #     sim_matrix_np = sim_matrix.cpu().numpy()
    #     adj_sorted = adj_np[sorted_idx][:, sorted_idx]
    #     sim_matrix_sorted = sim_matrix_np[sorted_idx][:, sorted_idx]

    #     plot_heatmap(sim_matrix_sorted, title=f"Similarity_Matrix_{tag}")
    #     plot_heatmap(adj_sorted, title=f"Adjacency_Matrix_{tag}")

    def forward(self):
        if self.feature_mode == "schema":
            features = self.sample_synthetic_features(self.num_nodes)
            _, _, edge_index = self.get_adj_wo_labels(features)
            return features, edge_index

        sim_matrix, adj, edge_index = self.get_adjacency_by_similarity()
        # self.show_graph_heatmap(sim_matrix, adj)
        # z = torch.randn(1, 64).to(self.device)
        # features = self.feature_gen(z).view(self.num_nodes, self.feature_dim)
        # sim_matrix, adj, edge_index = self.get_adj_wo_labels(features)
        # return features, edge_index
        return self.node_features, edge_index
    
    def random_graph(self):
        if self.feature_mode == "schema":
            rand_features = self.sample_synthetic_features(self.num_nodes)
            _, _, edge_index = self.get_adj_wo_labels(rand_features)
        else:
            # Re-sample the paper's multi-class CSBM on every training step:
            # uniform latent labels, class-conditioned Gaussian features, and
            # separate k_intra/k_inter similarity neighbours.
            rand_labels = torch.randint(
                self.class_num,
                (self.num_nodes,),
                device=self.device,
            )
            rand_features = self.sample_grap3_features(rand_labels)
            _, _, edge_index = self.get_adjacency_by_similarity(
                rand_features,
                rand_labels,
            )
        return rand_features, edge_index
