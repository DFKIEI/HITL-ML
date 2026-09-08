#!/usr/bin/env python3

import argparse
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_dataloaders
from helpers_batch import unpack_batch
from helpers_latent import compute_latents
from helpers_llm import LLMState, update_llm_state
from helpers_training import evaluate
from helpers_viz import save_latent_visualization
from models import build_model


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def should_run_epoch(epoch: int, interval: int, early_epochs: int) -> bool:
    if epoch <= max(0, early_epochs):
        return True
    if interval <= 0:
        return False
    return (epoch % interval) == 0




def main():
    parser = argparse.ArgumentParser()

    training_group = parser.add_argument_group("Training (Cross-Entropy)")
    training_group.add_argument("--data-dir", type=str, default="./data")
    training_group.add_argument(
        "--dataset", type=str, choices=["cifar10", "pamap2", "speechcommands15"], default="pamap2"
    )
    training_group.add_argument("--model", type=str, default="")
    training_group.add_argument("--epochs", type=int, default=30)
    training_group.add_argument("--batch-size", type=int, default=128)
    training_group.add_argument("--num-workers", type=int, default=2)
    training_group.add_argument("--lr", type=float, default=1e-3)
    training_group.add_argument("--latent-dim", type=int, default=128)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument(
        "--early-save-epochs",
        type=int,
        default=5,
        help="Always run/save LLM+viz for epochs 1..N, then use interval flags.",
    )
    # Device is auto-selected: CUDA -> MPS -> CPU.
    training_group.add_argument(
        "--mode",
        type=str,
        choices=["ce", "llm_extract"],
        default="ce",
        help="ce=plain cross-entropy, llm_extract=LLM suggestions + 2D viz.",
    )

    llm_group = parser.add_argument_group("LLM Extraction (Suggestions)")
    llm_group.add_argument("--llm-interval", type=int, default=5)
    llm_group.add_argument("--llm-q-very-close", type=float, default=0.1)
    llm_group.add_argument("--llm-q-close", type=float, default=0.3)
    llm_group.add_argument("--llm-q-far", type=float, default=0.7)
    llm_group.add_argument("--llm-q-very-far", type=float, default=0.9)
    llm_group.add_argument("--llm-eval-samples", type=int, default=10000)

    llm_group.add_argument("--llm-model", type=str, default="gpt-5-mini")
    llm_group.add_argument("--llm-temperature", type=float, default=0.2)
    llm_group.add_argument("--llm-max-output-tokens", type=int, default=1000)
    
    llm_group.add_argument("--llm-max-constraints", type=int, default=10)
    llm_group.add_argument("--llm-max-pairs", type=int, default=100)
    llm_group.add_argument("--llm-debug", action="store_true")

    viz_group = parser.add_argument_group("Latent Visualization")
    viz_group.add_argument("--viz-enabled", action="store_true")
    viz_group.add_argument("--viz-method", type=str, choices=["tsne", "umap", "pca"], default="umap")
    viz_group.add_argument("--viz-interval", type=int, default=5)
    viz_group.add_argument("--viz-out-dir", type=str, default="./results")

    args = parser.parse_args()
    set_seed(args.seed)

    do_llm = args.mode == "llm_extract"
    if args.mode == "llm_extract":
        args.viz_enabled = True

    train_loader, test_loader, eval_loader, dataset_name, class_names = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_samples=args.llm_eval_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    model_name = args.model.strip()
    if not model_name:
        if args.dataset == "cifar10":
            model_name = "CNN"
        elif args.dataset == "pamap2":
            model_name = "tscnn"
        else:
            model_name = "audiocnn"

    sample_batch = next(iter(train_loader))
    sample_x, _ = unpack_batch(sample_batch)

    if isinstance(sample_x, torch.Tensor):
        if sample_x.dim() == 4:
            input_channels = sample_x.size(1)
        elif sample_x.dim() == 3:
            input_channels = min(sample_x.size(1), sample_x.size(2))
        else:
            input_channels = 3
    else:
        input_channels = 3

    model = build_model(
        model_name,
        latent_dim=args.latent_dim,
        num_classes=len(class_names),
        input_channels=input_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    llm_state = LLMState(
        t_very_close=0.3,
        t_close=0.6,
        t_far=1.2,
        t_very_far=1.5,
        constraints=[],
        num_classes=len(class_names),
        class_names=class_names,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            images, targets = unpack_batch(batch)
            images = images.to(device)
            targets = targets.to(device)

            logits, z = model(images, return_latent=True)
            ce_loss = F.cross_entropy(logits, targets)
            llm_loss = torch.tensor(0.0, device=device)
            loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()

        acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={total_loss/len(train_loader):.4f} "
            f"ce={total_ce_loss/len(train_loader):.4f} "
            "llm=0.0000 "
            f"acc={acc:.4f}"
        )

        run_llm_epoch = do_llm and should_run_epoch(
            epoch=epoch,
            interval=args.llm_interval,
            early_epochs=args.early_save_epochs,
        )
        if run_llm_epoch:
            prompt_dump_path = None
            response_dump_path = None
            if args.viz_enabled:
                base_name = f"{dataset_name}_{model_name}_epoch{epoch}"
                prompt_dump_path = f"{args.viz_out_dir}/{base_name}_prompt.json"
                response_dump_path = f"{args.viz_out_dir}/{base_name}_response.json"
            llm_state = update_llm_state(
                model=model,
                loader=eval_loader,
                device=device,
                max_samples=args.llm_eval_samples,
                q_very_close=args.llm_q_very_close,
                q_close=args.llm_q_close,
                q_far=args.llm_q_far,
                q_very_far=args.llm_q_very_far,
                llm_model=args.llm_model,
                llm_temperature=args.llm_temperature,
                llm_max_output_tokens=args.llm_max_output_tokens,
                llm_max_constraints=args.llm_max_constraints,
                llm_max_pairs=args.llm_max_pairs,
                llm_debug=args.llm_debug,
                num_classes=len(class_names),
                class_names=class_names,
                dataset_name=dataset_name,
                prompt_dump_path=prompt_dump_path,
                response_dump_path=response_dump_path,
            )
            print(
                f"[Epoch {epoch}] LLM suggestions: {llm_state.suggestion_count} "
                f"(t_vclose={llm_state.t_very_close:.3f}, "
                f"t_close={llm_state.t_close:.3f}, "
                f"t_far={llm_state.t_far:.3f}, "
                f"t_vfar={llm_state.t_very_far:.3f})"
            )

        run_viz_epoch = args.viz_enabled and should_run_epoch(
            epoch=epoch,
            interval=args.viz_interval,
            early_epochs=args.early_save_epochs,
        )
        if run_viz_epoch:
            z_vis, y_vis = compute_latents(
                model=model,
                loader=eval_loader,
                device=device,
                max_samples=args.llm_eval_samples,
            )
            save_latent_visualization(
                z=z_vis,
                y=y_vis,
                dataset_name=dataset_name,
                model_name=model_name,
                epoch=epoch,
                method=args.viz_method,
                out_dir=args.viz_out_dir,
                class_names=class_names,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
