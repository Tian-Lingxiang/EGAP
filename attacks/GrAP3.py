import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml

from attacks.attsrc.GrAPattacker import GrAPattacker
from attacks.attsrc.graphgenerator import GraphGenerator
from attacks.attsrc.prompt import GraphPrompt
from src.graphmodel import *


def load_dataset_prompt(dataset_name):
    prompt_path = Path(__file__).resolve().parents[1] / "dataset_prompts.yaml"
    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        prompts = yaml.safe_load(prompt_file)
    for name, prompt in prompts.items():
        if name.lower() == dataset_name.lower():
            return prompt
    raise KeyError(f"No dataset prompt found for {dataset_name!r} in {prompt_path}")


def extract_by_grap3(
    args,
    pre_model,
    target_model,
    target_data,
    target_model_name,
    device,
    seed,
    if_prompt=True,
):

    print("\n[GrAP3 Attack]")
    dataset_prompt = load_dataset_prompt(args.target_dataset)
    target_class_num = int(dataset_prompt["num_classes"])
    target_feature_dim = int(dataset_prompt["feature_dim"])
    actual_class_num = int(target_data.y.max().item()) + 1
    actual_feature_dim = int(target_data.x.shape[1])
    if (target_class_num, target_feature_dim) != (
        actual_class_num,
        actual_feature_dim,
    ):
        raise ValueError(
            "Dataset prompt schema does not match the target model data: "
            f"prompt=({target_class_num} classes, {target_feature_dim} features), "
            f"actual=({actual_class_num} classes, {actual_feature_dim} features)"
        )

    surrogate_model = prepare_finetune_model(
        pre_model, target_class_num, target_feature_dim, device
    )
    generator_num_nodes = 500
    graph_generator = GraphGenerator(
        generator_num_nodes,
        target_feature_dim,
        target_class_num,
        device,
        dataset_prompt=dataset_prompt,
        feature_mode="grap3",
    ).to(device)
    prompt_nodes_num = 10
    attack_prompt = GraphPrompt(
        prompt_nodes_num=prompt_nodes_num,
        feature_dim=target_feature_dim,
        inner_prune=0.4,
        cross_prune=0.52,
        device=device,
        # The published GrAP3 construction uses dense Kaiming-initialized
        # prompt features and thresholded cross edges.
        reference_features=None,
        cross_top_k=None,
    ).to(device)
    # Attack setup
    attacker = GrAPattacker(
        args, target_model, surrogate_model, graph_generator, attack_prompt
    )
    if if_prompt:
        surrogate_model, best_query_fidelity = attacker.attack_prompt()
        # Prompt nodes are used only to enhance synthetic extraction queries.
        # Evaluate the deployed surrogate on the original, unprompted test graph.
        surrogate_acc, target_acc, fidelity, f1 = attacker.evaluate_attack(target_data)
        # surrogate_acc, target_acc, fidelity, f1 = (
        #     attacker.evaluate_attack_batch(target_data)
        # )
    else:
        surrogate_model, best_query_fidelity = attacker.attack()
        surrogate_acc, target_acc, fidelity, f1 = attacker.evaluate_attack(target_data)
    # save the results

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    final_info = (
        f"current_time:{current_time}\n"
        f"attack_method:GrAP3\n"
        f"random_seed:{seed}\n"
        f"if_prompt:{if_prompt}\n"
        f"pretrain_method:{args.pretrain_method}\n"
        f"target_model_name:{target_model_name}\n"
        f"target_acc:{target_acc:.4f}\tbest_query_fidelity:{best_query_fidelity:.4f}\t"
        f"best_query_balanced_fidelity:{attacker.best_query_metrics['balanced_fidelity']:.4f}\t"
        f"best_query_coverage:{attacker.best_query_metrics['coverage']:.4f}\t"
        f"surrogate_acc:{surrogate_acc:.4f}\n"
        f"fidelity:{fidelity:.4f}\tf1_score:{f1:.4f}\n\n"
    )
    print(final_info)

    SURROGATE_SAVE_PATH = os.path.join(
        "./saved_models", target_model_name, "surrogate_model"
    )
    os.makedirs(SURROGATE_SAVE_PATH, exist_ok=True)
    surrogate_save_name = f"grap3_{seed}.pt"
    surrogate_save_full_path = os.path.join(SURROGATE_SAVE_PATH, surrogate_save_name)
    torch.save(surrogate_model, surrogate_save_full_path)

    return surrogate_model
