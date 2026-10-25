"""在模型沙盒中构建 GNNStealing Type I.2 窃取模型。"""

import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

from attacks.attsrc.GNNStealingattacker import (
    GNNStealingModel, query_subgraph, target_response, train_gnnstealing,
)
from src.detection.graph_io import model_schema
from src.detection.sandbox import file_sha256


def extract_by_gnnstealing(
    args, target_model, target_data, target_model_name, device, seed,
    *, output_root="saved_models",
):
    """使用已知特征、边和查询标签，从随机初始化训练 Type I.2。"""
    if not target_data.test_mask.any():
        raise ValueError("GNNStealing 需要非空测试分区")
    input_dim, output_dim = model_schema(target_model)
    if input_dim != target_data.x.shape[1] or output_dim is None:
        raise ValueError("GNNStealing 数据与 Target schema 不匹配")
    query, nodes = query_subgraph(
        target_data, args.gnnstealing_query_ratio, seed,
    )
    directory = Path(output_root) / target_model_name / "surrogate_model"
    path = directory / f"gnnstealing_{seed}.pt"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"GNNStealing 输出已存在：{path}")
    torch.manual_seed(seed)
    model = GNNStealingModel(
        input_dim, args.hidden_dim, output_dim, args.model_type,
    ).to(device)
    if getattr(target_model, "class_names", None) is not None:
        model.class_names = tuple(target_model.class_names)
    training = train_gnnstealing(
        model, target_model, query, epochs=args.num_epochs, seed=seed,
    )

    # 与其他沙盒攻击一致，在完整原图的测试掩码上报告指标。
    mask = target_data.test_mask
    with torch.no_grad():
        predicted = model(
            target_data.x, target_data.edge_index,
        )[mask].argmax(1).cpu()
    expected = target_response(target_model, target_data)[mask].argmax(1).cpu()
    truth = target_data.y.reshape(-1)[mask].cpu()
    metrics = {
        "surrogate_accuracy": accuracy_score(truth, predicted),
        "target_accuracy": accuracy_score(truth, expected),
        "fidelity": accuracy_score(expected, predicted),
        "f1": f1_score(
            expected, predicted, average="weighted", zero_division=0,
        ),
    }
    metadata = {
        "attack_method": "GNNStealing", "variant": "Type I.2",
        "reference": "https://github.com/xinleihe/GNNStealing",
        "target_model_name": target_model_name,
        "schema": {"feature_dim": input_dim, "num_classes": output_dim},
        "response": "posterior_probability", "seed": seed,
        "initialization": "random", "model_type": args.model_type,
        "hidden_dim": args.hidden_dim, "epochs": args.num_epochs,
        "encoder_learning_rate": 0.001,
        "classifier": {
            "type": "MLPClassifier", "hidden_dim": 100, "max_iter": 300,
        },
        "query_ratio": args.gnnstealing_query_ratio,
        "query_graph": "induced_subgraph_of_training_partition",
        "query_labels": "ground_truth",
        "query_nodes": nodes.cpu().tolist(),
        "validation_nodes": (
            target_data.val_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        ),
        "test_nodes": mask.nonzero(as_tuple=True)[0].cpu().tolist(),
        "target_training_overlap": "query_nodes_are_target_training_nodes",
        "evaluation": "full_graph_test_mask",
        "evaluation_target_calls": 1,
        "evaluation_node_responses": target_data.num_nodes,
        **training, "metrics": metrics,
    }
    provenance = getattr(target_data, "attack_split_provenance", None)
    if provenance is not None:
        metadata["split_provenance"] = provenance
        metadata["target_training_overlap"] = (
            "query_nodes_are_reconstructed_training_nodes"
        )
    target_path = Path("saved_models") / target_model_name / "target_model.pt"
    if target_path.is_file():
        metadata["target_sha256"] = file_sha256(target_path)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    metadata["model_sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"GNNStealing 已保存到 {path}，测试指标：{metrics}")
    return model
