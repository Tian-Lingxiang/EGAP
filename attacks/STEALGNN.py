"""构建、评估并保存 STEALGNN Type II 窃取模型。"""

from dataclasses import asdict
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

from attacks.attsrc.STEALGNNattacker import (
    STEALGNNConfig, STEALGNNGenerator, STEALGNNModel,
    query_hard_labels, train_stealgnn,
)
from src.detection.graph_io import model_schema
from src.detection.sandbox import file_sha256


def extract_by_stealgnn(
    args, target_model, target_data, target_model_name, device, seed,
    *, output_root="saved_models",
):
    """使用公开 schema 和合成图提取模型，在训练结束后评估真实测试图。"""
    input_dim, output_dim = model_schema(target_model)
    if (
        input_dim is None or input_dim < 1
        or output_dim is None or output_dim < 2
    ):
        raise ValueError("STEALGNN 需要 Target 的输入维度和输出类别数")
    if input_dim != target_data.x.shape[1]:
        raise ValueError("STEALGNN 评估数据与 Target schema 不匹配")
    mask = target_data.test_mask
    if (
        mask.dtype != torch.bool
        or mask.shape != (target_data.num_nodes,)
        or not mask.any()
    ):
        raise ValueError("STEALGNN 需要非空的一维布尔测试掩码")
    directory = Path(output_root) / target_model_name / "surrogate_model"
    path = directory / f"stealgnn_{seed}.pt"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"STEALGNN 输出已存在：{path}")
    config = STEALGNNConfig(
        num_nodes=args.stealgnn_num_nodes,
        noise_dim=args.stealgnn_noise_dim,
        threshold=args.stealgnn_threshold,
    )
    torch.manual_seed(seed)
    model = STEALGNNModel(
        input_dim, args.hidden_dim, output_dim, args.model_type,
    ).to(device)
    if getattr(target_model, "class_names", None) is not None:
        model.class_names = tuple(target_model.class_names)
    generator = STEALGNNGenerator(input_dim, config).to(device)
    training = train_stealgnn(
        model, target_model, generator, epochs=args.num_epochs,
    )

    # 使用最终轮代理；真实图与标签只在此处参与训练后的评估。
    with torch.no_grad():
        output = model(target_data.x, target_data.edge_index)
        predicted = output[mask].argmax(1).cpu()
    expected = query_hard_labels(
        target_model, target_data.x, target_data.edge_index, output_dim,
    )[mask].cpu()
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
        "attack_method": "STEALGNN", "variant": "Type II",
        "reference": (
            "https://www.usenix.org/system/files/usenixsecurity24-zhuang.pdf"
        ),
        "target_model_name": target_model_name, "seed": seed,
        "schema": {"feature_dim": input_dim, "num_classes": output_dim},
        "initialization": "random", "response": "hard_label",
        "model_type": args.model_type, "hidden_dim": args.hidden_dim,
        "num_layers": 2, "dropout": 0.5, "epochs": args.num_epochs,
        "graph_structure": "cosine_threshold", "config": asdict(config),
        "checkpoint_selection": "final_epoch",
        "evaluation": "full_graph_test_mask",
        "evaluation_target_calls": 1,
        "evaluation_node_responses": target_data.num_nodes,
        "test_nodes": mask.nonzero(as_tuple=True)[0].cpu().tolist(),
        **training, "metrics": metrics,
    }
    provenance = getattr(target_data, "attack_split_provenance", None)
    if provenance is not None:
        metadata["split_provenance"] = provenance
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
    print(f"STEALGNN 已保存到 {path}，测试指标：{metrics}")
    return model
