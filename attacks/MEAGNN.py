import os
from datetime import datetime

import torch

from attacks.attsrc.MEAattacker import MEAattacker
from src.graphmodel import NodeIdentityAdapter, evaluate_model, prepare_finetune_model


def extract_by_meagnn(
    args, pre_model, target_model, target_data, target_model_name, device, seed
):
    
    print("\n[MEAGNN Attack-2]")
    target_class_num = target_data.y.max().item() + 1

    # Keep the public input schema so callers can pass the original dataset.
    # Internally, Attack-2 represents each real node by identity, exactly as
    # Linear(I_N) in the reference implementation, without reading x values.
    surrogate_model = prepare_finetune_model(
        pre_model, target_class_num, target_data.x.shape[1], device
    ).to(device)
    surrogate_model.adapter = NodeIdentityAdapter(
        num_nodes=target_data.num_nodes,
        output_dim=768,
        input_dim=target_data.x.shape[1],
    ).to(device)
    # The reference Attack-2 trains the whole duplicated GNN.  Keeping the
    # transferred encoder frozen prevents it from adapting to identity inputs.
    for parameter in surrogate_model.parameters():
        parameter.requires_grad = True

    attacker = MEAattacker(args, target_model, surrogate_model, target_data)

    surrogate_model, best_eval_fidelity = attacker.attack()
    # As with GrAP, report accuracy and fidelity on the real target dataset.
    # Synthetic inputs are used only for extraction; both models receive the
    # same real features and graph here.
    surrogate_acc, target_acc, fidelity, f1 = attacker.evaluate_attack(target_data)


    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    final_info = (
        f"current_time:{current_time}\n"
        f"attack_method:MEAGNN(Attack-2)\n"
        f"random_seed:{seed}\n"
        f"pretrain_method:{args.pretrain_method}\n"
        f"target_model_name:{target_model_name}\n"
        f"target_acc:{target_acc:.4f}\t"
        f"best_synthetic_eval_fidelity:{best_eval_fidelity:.4f}\t"
        f"surrogate_acc:{surrogate_acc:.4f}\n"
        f"fidelity:{fidelity:.4f}\tf1_score:{f1:.4f}\n\n"
    )
    print(final_info)

    SURROGATE_SAVE_PATH = os.path.join(
        "./saved_models", target_model_name, "surrogate_model"
    )
    os.makedirs(SURROGATE_SAVE_PATH, exist_ok=True)
    surrogate_save_name = f"meagnn_{seed}.pt"
    surrogate_save_full_path = os.path.join(SURROGATE_SAVE_PATH, surrogate_save_name)
    torch.save(surrogate_model, surrogate_save_full_path)

    return surrogate_model
