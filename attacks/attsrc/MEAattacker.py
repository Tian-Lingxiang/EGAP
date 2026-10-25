import copy
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


class MEAattacker:
    # Attack-2-specific defaults live here instead of polluting model_build's
    # common command-line interface.
    ATTACK_RATIO = 0.25
    EVAL_EVERY = 10
    LEARNING_RATE = 2e-2
    WEIGHT_DECAY = 5e-4

    def __init__(
        self,
        args,
        target_model,
        surrogate_model,
        target_dataset,
    ):
        self.args = args
        self.target_model = target_model
        self.surrogate_model = surrogate_model
        self.target_dataset = target_dataset
        self.num_epochs = args.num_epochs
        self.eval_every = self.EVAL_EVERY
        
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        
        # Attributes are unavailable to the attacker.  This fixed placeholder
        # carries no private information and needs no training; the surrogate's
        # identity adapter derives the original Attack-2 representation solely
        # from the ordering of these real nodes.
        self.syn_features = torch.zeros(
            target_dataset.num_nodes,
            target_dataset.x.shape[1],
            dtype=next(self.surrogate_model.parameters()).dtype,
            device=self.device,
        )
        self.known_edge_index = target_dataset.edge_index

        self.loss_function = nn.NLLLoss()

        query_count = max(1, int(target_dataset.num_nodes * self.ATTACK_RATIO))
        # Draw attack nodes from the training partition so that validation
        # accuracy is measured on nodes never used to fit the surrogate.
        query_candidates = target_dataset.train_mask.nonzero(as_tuple=False).view(-1)
        if query_count > query_candidates.numel():
            raise ValueError("Not enough training nodes for the MEAGNN query budget")
        permutation = torch.randperm(query_candidates.numel(), device=self.device)
        query_nodes = query_candidates[permutation[:query_count]]
        self.query_mask = torch.zeros(
            target_dataset.num_nodes, dtype=torch.bool, device=self.device
        )
        self.query_mask[query_nodes] = True

        self.eval_mask = target_dataset.test_mask
        if not self.eval_mask.any():
            raise ValueError("No test nodes available for MEAGNN evaluation")

        print(
            "Attack-2 setup: "
            f"nodes={target_dataset.num_nodes}, queries={query_count} "
            f"validation nodes={target_dataset.val_mask.sum().item()}, "
            f"test nodes={self.eval_mask.sum().item()}"
        )
        
        # 优化器配置
        self.surrogate_optimizer = torch.optim.Adam(
            surrogate_model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

    def _validate_evaluation_dataset(self, dataset):
        expected_input_dim = self.surrogate_model.adapter.input_dim
        if dataset.x.shape[1] != expected_input_dim:
            raise ValueError(
                "MEAGNN evaluation features do not match the surrogate input schema"
            )

    def attack(self):
        self.surrogate_model.train()
        self.target_model.eval()
        
        best_eval_fidelity = -1.0
        best_epoch = 0
        best_state = copy.deepcopy(self.surrogate_model.state_dict())
        
        # The attacker queries real node IDs.  The victim resolves those IDs
        # using its private attributes internally; only its hard labels leave
        # the service.  The surrogate itself never sees the private features
        # during extraction and is trained with the fixed synthetic matrix.
        with torch.no_grad():
            target_output = self.target_model(
                self.target_dataset.x, self.known_edge_index
            )
            target_labels = target_output.argmax(dim=1)
            
        for epoch in range(self.num_epochs):
            tic = time.time()
            self.surrogate_optimizer.zero_grad()
            
            surrogate_output = self.surrogate_model(self.syn_features, self.known_edge_index)
            
            loss = self.loss_function(
                surrogate_output[self.query_mask],
                target_labels[self.query_mask],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.surrogate_model.parameters(), max_norm=1.0)
            self.surrogate_optimizer.step()
            
            toc = time.time()
            print(f"Epoch:{epoch}, Time(s):{toc - tic:.2f}, Surrogate Loss: {loss.item():.4f}")
            
            if ((epoch + 1) % self.eval_every == 0) or (epoch == self.num_epochs - 1):
                self.surrogate_model.eval()
                with torch.no_grad():
                    eval_pred_output = self.surrogate_model(self.syn_features, self.known_edge_index)
                    
                eval_fidelity = accuracy_score(
                    target_labels[self.target_dataset.val_mask].cpu(),
                    eval_pred_output[self.target_dataset.val_mask].argmax(dim=1).cpu(),
                )
                print(f"Synthetic Eval Fidelity {eval_fidelity:.4f}")
                if eval_fidelity > best_eval_fidelity:
                    best_eval_fidelity = eval_fidelity
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.surrogate_model.state_dict())
                self.surrogate_model.train()
                print(
                    f"Best Synthetic Eval Fidelity {best_eval_fidelity:.4f} "
                    f"at Epoch {best_epoch}"
                )
        self.surrogate_model.load_state_dict(best_state)
        return self.surrogate_model, best_eval_fidelity

    def evaluate_attack(self, evaluation_dataset):
        """Evaluate on real dataset nodes using fixed synthetic node features.

        Both the target and surrogate receive the same real features and graph;
        synthetic features are restricted to extraction/training.
        """
        self._validate_evaluation_dataset(evaluation_dataset)
        evaluation_edge_index = evaluation_dataset.edge_index
        evaluation_mask = evaluation_dataset.test_mask
        if not evaluation_mask.any():
            raise ValueError("No test nodes available for MEAGNN evaluation")

        self.surrogate_model.eval()
        self.target_model.eval()
        with torch.no_grad():
            surrogate_out = self.surrogate_model(
                evaluation_dataset.x, evaluation_edge_index
            )
            target_out = self.target_model(
                evaluation_dataset.x, evaluation_edge_index
            )
            surrogate_pred = surrogate_out[evaluation_mask].argmax(dim=1)
            target_pred = target_out[evaluation_mask].argmax(dim=1)
            
        surrogate_acc = accuracy_score(
            evaluation_dataset.y.squeeze()[evaluation_mask].cpu(),
            surrogate_pred.cpu(),
        )
        target_acc = accuracy_score(
            evaluation_dataset.y.squeeze()[evaluation_mask].cpu(),
            target_pred.cpu(),
        )
        fidelity = accuracy_score(surrogate_pred.cpu(), target_pred.cpu())
        f1 = f1_score(target_pred.cpu(), surrogate_pred.cpu(), average="weighted")

        return surrogate_acc, target_acc, fidelity, f1
