import copy
import math
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import NeighborLoader


class GrAPattacker:
    def __init__(
        self,
        args,
        target_model,
        surrogate_model,
        graph_generator,
        attack_prompt,
    ):
        self.model_type = args.model_type
        self.target_model = target_model
        self.surrogate_model = surrogate_model
        self.graph_generator = graph_generator
        self.n_generator_steps = 2
        self.n_surrogate_steps = 5
        self.num_epochs = args.num_epochs
        self.eval_every = args.eval_every
        # Keep the response-matching objective used by the original GrAP3
        # implementation. GraphModel returns log probabilities, so this
        # matches the complete soft response rather than only hard labels.
        self.loss_function = nn.MSELoss()
        # self.generator_optimizer = torch.optim.Adam(
        #     self.graph_generator.parameters(), lr=args.generator_lr
        # )
        self.prompt = attack_prompt
        self.prompt_optimizer = torch.optim.Adam(
            self.prompt.parameters(), lr=args.prompt_lr
        )
        encoder_params = list(surrogate_model.convs[-1].parameters())
        adapter_params = list(surrogate_model.adapter.parameters())
        classifier_params = list(surrogate_model.classifier.parameters())
        self.surrogate_optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": classifier_params, "lr": args.classifier_lr},
                {"params": adapter_params, "lr": args.classifier_lr},
            ],
        )
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.best_query_metrics = self.empty_query_metrics()

    @staticmethod
    def empty_query_metrics():
        return {
            "fidelity": 0.0,
            "balanced_fidelity": 0.0,
            "coverage": 0.0,
            "label_entropy": 0.0,
            "selection_score": 0.0,
        }

    @staticmethod
    def query_metrics(surrogate_output, target_output):
        """Measure agreement without rewarding single-class query collapse."""
        target_pred = target_output.argmax(dim=1)
        surrogate_pred = surrogate_output.argmax(dim=1)
        num_classes = target_output.shape[1]
        present_classes = target_pred.unique()

        fidelity = (surrogate_pred == target_pred).float().mean().item()
        per_class_fidelity = []
        for class_index in present_classes:
            class_mask = target_pred == class_index
            per_class_fidelity.append(
                (surrogate_pred[class_mask] == target_pred[class_mask])
                .float()
                .mean()
            )
        balanced_fidelity = torch.stack(per_class_fidelity).mean().item()
        coverage = present_classes.numel() / num_classes

        class_histogram = torch.bincount(
            target_pred, minlength=num_classes
        ).to(dtype=torch.float32)
        class_probabilities = class_histogram / class_histogram.sum()
        nonzero_probabilities = class_probabilities[class_probabilities > 0]
        label_entropy = -(
            nonzero_probabilities * nonzero_probabilities.log()
        ).sum().item()
        if num_classes > 1:
            label_entropy /= math.log(num_classes)

        # A constant target response has zero entropy and therefore can never
        # look like a perfect validation query merely because both models
        # predict the same majority class.
        selection_score = balanced_fidelity * label_entropy
        return {
            "fidelity": fidelity,
            "balanced_fidelity": balanced_fidelity,
            "coverage": coverage,
            "label_entropy": label_entropy,
            "selection_score": selection_score,
        }

    @staticmethod
    def query_node_output(model_output, num_query_nodes):
        """Exclude auxiliary prompt nodes as required by Eq. (18)."""
        if not 0 < num_query_nodes <= model_output.shape[0]:
            raise ValueError("num_query_nodes must index a non-empty output prefix")
        return model_output[:num_query_nodes]

    def train_generator(self):
        self.graph_generator.train()
        self.surrogate_model.eval()
        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            features, edge_index = self.graph_generator()
            with torch.no_grad():
                target_output = self.target_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)
            loss = self.loss_function(surrogate_output, target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.graph_generator.parameters(), max_norm=1.0
            )
            self.generator_optimizer.step()
            total_loss += loss.item()
        return total_loss / self.n_generator_steps

    def train_prompt(self):
        self.prompt.train()
        self.surrogate_model.eval()
        total_loss = 0
        surrogate_requires_grad = [
            parameter.requires_grad for parameter in self.surrogate_model.parameters()
        ]
        for parameter in self.surrogate_model.parameters():
            parameter.requires_grad_(False)
        for _ in range(self.n_generator_steps):
            self.prompt_optimizer.zero_grad()
            features, edge_index = self.graph_generator()
            features = features.detach()
            now_features, now_edge_index = self.prompt(features, edge_index)
            with torch.no_grad():
                target_output = self.target_model(now_features, now_edge_index)
            surrogate_output = self.surrogate_model(now_features, now_edge_index)
            target_output = self.query_node_output(target_output, features.shape[0])
            surrogate_output = self.query_node_output(
                surrogate_output, features.shape[0]
            )
            loss = self.loss_function(surrogate_output, target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prompt.parameters(), max_norm=1.0)
            self.prompt_optimizer.step()
            self.prompt.project_features()
            total_loss += loss.item()
        for parameter, requires_grad in zip(
            self.surrogate_model.parameters(), surrogate_requires_grad
        ):
            parameter.requires_grad_(requires_grad)
        return total_loss / self.n_generator_steps

    def train_surrogate(self):
        self.surrogate_model.train()
        # self.graph_generator.eval()
        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            # features, edge_index = self.graph_generator()
            # print("random graph")
            features, edge_index = self.graph_generator.random_graph()
            with torch.no_grad():
                target_output = self.target_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)
            loss = self.loss_function(surrogate_output, target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.surrogate_model.parameters(), max_norm=1.0
            )
            self.surrogate_optimizer.step()
            total_loss += loss.item()
        return total_loss / self.n_surrogate_steps

    def train_surrogate_prompt(self):
        self.surrogate_model.train()
        self.prompt.eval()
        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            # Fresh multi-class CSBM graphs retain the paper's label-aware
            # distribution while providing independent query directions.
            features, edge_index = self.graph_generator.random_graph()
            now_features, now_edge_index = self.prompt(features, edge_index)
            now_features = now_features.detach()
            with torch.no_grad():
                target_output = self.target_model(now_features, now_edge_index)
            surrogate_output = self.surrogate_model(now_features, now_edge_index)
            target_output = self.query_node_output(target_output, features.shape[0])
            surrogate_output = self.query_node_output(
                surrogate_output, features.shape[0]
            )
            loss = self.loss_function(surrogate_output, target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.surrogate_model.parameters(), max_norm=1.0
            )
            self.surrogate_optimizer.step()
            total_loss += loss.item()
        return total_loss / self.n_surrogate_steps

    @staticmethod
    def is_better_query(metrics, query_loss, best_metrics, best_query_loss):
        candidate = (
            metrics["selection_score"],
            metrics["coverage"],
            metrics["balanced_fidelity"],
            metrics["fidelity"],
            -query_loss,
        )
        incumbent = (
            best_metrics["selection_score"],
            best_metrics["coverage"],
            best_metrics["balanced_fidelity"],
            best_metrics["fidelity"],
            -best_query_loss,
        )
        return candidate > incumbent

    def attack(self):
        avg_epoch_time = 0
        best_query_loss = float("inf")
        best_epoch = 0
        best_metrics = self.empty_query_metrics()
        best_state = copy.deepcopy(self.surrogate_model.state_dict())
        query_val_features, query_val_edge_index = self.graph_generator()

        for epoch in range(self.num_epochs):
            tic = time.time()
            surrogate_loss = self.train_surrogate()
            toc = time.time()
            avg_epoch_time += toc - tic
            print(
                f"Epoch:{epoch}, Epoch Time(s):{toc - tic:.2f}, "
                f"Surrogate Loss: {surrogate_loss:.4f}"
            )
            if ((epoch + 1) % self.eval_every == 0) or (epoch == self.num_epochs - 1):
                self.surrogate_model.eval()
                self.target_model.eval()
                with torch.no_grad():
                    surrogate_output = self.surrogate_model(
                        query_val_features, query_val_edge_index
                    )
                    target_output = self.target_model(
                        query_val_features, query_val_edge_index
                    )
                query_loss = self.loss_function(surrogate_output, target_output).item()
                metrics = self.query_metrics(surrogate_output, target_output)
                self.print_query_metrics(query_loss, metrics)
                if self.is_better_query(
                    metrics, query_loss, best_metrics, best_query_loss
                ):
                    best_metrics = metrics
                    best_query_loss = query_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.surrogate_model.state_dict())
                self.print_best_query(best_metrics, best_query_loss, best_epoch)

        print(f"Average Epoch Time(s): {avg_epoch_time / self.num_epochs:.2f}")
        self.surrogate_model.load_state_dict(best_state)
        self.best_query_metrics = best_metrics
        return self.surrogate_model, best_metrics["fidelity"]

    @staticmethod
    def print_query_metrics(query_loss, metrics):
        print(
            f"Query Val Loss {query_loss:.4f}, "
            f"Fidelity {metrics['fidelity']:.4f}, "
            f"Balanced Fidelity {metrics['balanced_fidelity']:.4f}, "
            f"Coverage {metrics['coverage']:.4f}, "
            f"Label Entropy {metrics['label_entropy']:.4f}, "
            f"Selection Score {metrics['selection_score']:.4f}"
        )

    @staticmethod
    def print_best_query(metrics, query_loss, epoch):
        print(
            f"Best Query Score {metrics['selection_score']:.4f}, "
            f"Fidelity {metrics['fidelity']:.4f}, "
            f"Coverage {metrics['coverage']:.4f}, "
            f"Loss {query_loss:.4f} at Epoch {epoch}"
        )

    def attack_prompt(self):
        avg_epoch_time = 0
        best_query_loss = float("inf")
        best_epoch = 0
        best_metrics = self.empty_query_metrics()
        best_surrogate_state = copy.deepcopy(self.surrogate_model.state_dict())
        best_prompt_state = copy.deepcopy(self.prompt.state_dict())
        # Fixed validation probes are synthetic and derived only from the public
        # dataset prompt. No private graph, features, masks, or labels are used
        # for attack-time model selection.
        query_val_features, query_val_edge_index = self.graph_generator()

        for epoch in range(self.num_epochs):
            tic = time.time()
            prompt_loss = self.train_prompt()
            sur_loss = self.train_surrogate_prompt()
            toc = time.time()
            avg_epoch_time += toc - tic
            print(
                f"Epoch:{epoch}, Epoch Time(s):{toc - tic:.2f}, Prompt Loss: {prompt_loss:.4f}, Surrogate Loss: {sur_loss:.4f}"
            )
            if ((epoch + 1) % self.eval_every == 0) or (epoch == self.num_epochs - 1):
                self.surrogate_model.eval()
                self.target_model.eval()
                with torch.no_grad():
                    prompted_features, prompted_edge_index = self.prompt(
                        query_val_features, query_val_edge_index
                    )
                    eval_pred_output = self.surrogate_model(
                        prompted_features,
                        prompted_edge_index,
                    )
                    eval_target_output = self.target_model(
                        prompted_features,
                        prompted_edge_index,
                    )
                eval_pred_output = self.query_node_output(
                    eval_pred_output, query_val_features.shape[0]
                )
                eval_target_output = self.query_node_output(
                    eval_target_output, query_val_features.shape[0]
                )
                query_loss = self.loss_function(
                    eval_pred_output, eval_target_output
                ).item()
                metrics = self.query_metrics(eval_pred_output, eval_target_output)
                self.print_query_metrics(query_loss, metrics)
                if self.is_better_query(
                    metrics, query_loss, best_metrics, best_query_loss
                ):
                    best_metrics = metrics
                    best_query_loss = query_loss
                    best_epoch = epoch
                    best_surrogate_state = copy.deepcopy(
                        self.surrogate_model.state_dict()
                    )
                    best_prompt_state = copy.deepcopy(self.prompt.state_dict())
                self.print_best_query(best_metrics, best_query_loss, best_epoch)

        avg_epoch_time /= self.num_epochs
        print(f"Average Epoch Time(s): {avg_epoch_time:.2f}")
        self.surrogate_model.load_state_dict(best_surrogate_state)
        self.prompt.load_state_dict(best_prompt_state)
        self.best_query_metrics = best_metrics
        return self.surrogate_model, best_metrics["fidelity"]

    def evaluate_attack(self, evaluation_dataset):
        self.surrogate_model.eval()
        self.target_model.eval()
        with torch.no_grad():
            surrogate_out = self.surrogate_model(
                evaluation_dataset.x, evaluation_dataset.edge_index
            )
            target_out = self.target_model(
                evaluation_dataset.x, evaluation_dataset.edge_index
            )
            surrogate_pred = surrogate_out[evaluation_dataset.test_mask].argmax(dim=1)
            target_pred = target_out[evaluation_dataset.test_mask].argmax(dim=1)
        surrogate_acc = accuracy_score(
            evaluation_dataset.y.squeeze()[evaluation_dataset.test_mask].cpu(),
            surrogate_pred.cpu(),
        )
        target_acc = accuracy_score(
            evaluation_dataset.y.squeeze()[evaluation_dataset.test_mask].cpu(),
            target_pred.cpu(),
        )
        fidelity = accuracy_score(surrogate_pred.cpu(), target_pred.cpu())
        f1 = f1_score(target_pred.cpu(), surrogate_pred.cpu(), average="weighted")

        return surrogate_acc, target_acc, fidelity, f1

    def evaluate_attack_batch(self, evaluation_dataset):
        # ===== Test Loader =====
        test_loader = NeighborLoader(
            evaluation_dataset,
            input_nodes=evaluation_dataset.test_mask,
            num_neighbors=[15, 10],  # 与训练 / 验证保持一致
            batch_size=4096,
            shuffle=False,
            num_workers=0,
        )

        self.surrogate_model.eval()
        self.target_model.eval()

        surrogate_preds = []
        target_preds = []
        gt_labels = []

        with torch.no_grad():
            for batch in test_loader:
                surrogate_out = self.surrogate_model(
                    batch.x,
                    batch.edge_index,
                )
                target_out = self.target_model(
                    batch.x,
                    batch.edge_index,
                )

                # 只取 target nodes
                surrogate_pred = surrogate_out[: batch.batch_size].argmax(dim=1)
                target_pred = target_out[: batch.batch_size].argmax(dim=1)

                surrogate_preds.append(surrogate_pred.cpu())
                target_preds.append(target_pred.cpu())
                gt_labels.append(batch.y[: batch.batch_size].squeeze().cpu())

        surrogate_preds = torch.cat(surrogate_preds, dim=0)
        target_preds = torch.cat(target_preds, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)

        surrogate_acc = accuracy_score(gt_labels, surrogate_preds)
        target_acc = accuracy_score(gt_labels, target_preds)
        fidelity = accuracy_score(target_preds, surrogate_preds)
        f1 = f1_score(target_preds, surrogate_preds, average="weighted")

        # print(
        #     f"Surrogate Acc: {surrogate_acc:.4f}, "
        #     f"Target Acc: {target_acc:.4f}, "
        #     f"Fidelity: {fidelity:.4f}, "
        #     f"F1: {f1:.4f}"
        # )

        return surrogate_acc, target_acc, fidelity, f1
