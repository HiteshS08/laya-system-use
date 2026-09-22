"""Kaggle 2xT4 DDP fine-tune of Laya. Adapted from Laya's typed-decisions notebook (Apache-2.0).

torchrun --standalone --nproc_per_node=2 train_ddp.py MODEL_DIR TRAIN_ITEMS.pt DEV_ITEMS.pt OUTPUT_DIR
"""

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from laya.common import build_model, proper_reward, temp_bucket
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer


def collate_train_batch(items, pad_id):
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items]),
    }


MIN_TEMP_SAMPLES = 10
DEV_CALIBRATION_CAP = 2000


def fit_one_temp(sel):
    if len(sel) < MIN_TEMP_SAMPLES:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    logits = torch.full((len(sel), kmax), -1e4)
    target = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        logits[i, : len(z)] = torch.tensor(z)
        target[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(target * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(torch.clamp(log_t.exp(), 0.1, 10.0).item())
    return temp if math.isfinite(temp) else 1.0  # nan/inf would reach the config and laya would clamp it to T=1e-3


def fit_temperatures(preds):
    """preds: (qtype, logits_1d, target_1d) tuples -> (per-qtype temps, {laya temp_bucket key: temp}).

    laya.agent prefers temperature_by_options[temp_bucket(qtype, k)] over the per-qtype temperature, so the
    buckets are what actually reach inference. Buckets with fewer than MIN_TEMP_SAMPLES are left out.
    """
    temps = [1.2, 1.2, 1.2]  # only `choice` questions exist in this project; score and noul keep the default
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    grouped = defaultdict(list)
    for qt, z, t in preds:
        grouped[temp_bucket(qt, len(z))].append((z, t))
    by_bucket = {b: fit_one_temp(sel) for b, sel in grouped.items() if len(sel) >= MIN_TEMP_SAMPLES}
    return temps, by_bucket


def write_config(cfg, output_dir, temps, by_bucket):
    """Write a loadable rl_agent_config.json; each call replaces the previous file, nothing is merged in.

    `by_bucket` is a fresh map: the base config's buckets were fitted for the base model and would shadow
    `temperature` in laya.agent.
    """
    out = {**cfg, "fine_tuned": True, "model_name": "laya-browser-mind2web",
           "temperature": list(temps), "temperature_by_options": dict(by_bucket)}
    Path(output_dir, "rl_agent_config.json").write_text(json.dumps(out, indent=2))


def calibrate(model, dev_items, pad_id, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(dev_items), 16):
            chunk = dev_items[start : start + 16]
            cb = collate_train_batch(chunk, pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                  cb["marker_pos"].to(device), cb["marker_mask"].to(device), cb["qtype"].to(device))
            arr = logits.float().cpu().numpy()
            for r, it in enumerate(chunk):
                preds.append((it["qtype"], arr[r, : len(it["markers"])], it["target"]))
    return fit_temperatures(preds)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model_dir, train_path, dev_path, output_dir = sys.argv[1:5]

    cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
    meta = json.loads(Path(train_path).with_suffix(".meta.json").read_text())
    cfg.update(gradient_checkpointing=True, max_tokens_per_batch=4096,
               max_len=meta["max_len"], head_max_len=meta["head_max_len"])  # must match how the items were tokenised

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(train_path, weights_only=False)
    per_rank = len(all_items) // world  # equal counts, so every rank takes the same number of optimiser steps
    my_items = all_items[rank::world][:per_rank]

    epochs = int(os.environ.get("EPOCHS", "4"))
    micro_batch, grad_accum, group_size = 8, 4, 4
    lr_encoder, lr_head, sigma_start, sigma_end = 2.5e-5, 1.0e-4, 0.4, 0.1
    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_encoder}, {"params": head_params, "lr": lr_head}], weight_decay=0.01)
    total_updates = (len(my_items) // (micro_batch * grad_accum)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if rank == 0:
        print(f"items={len(all_items)} per_rank={len(my_items)} epochs={epochs}", flush=True)
    t0 = time.time()

    for epoch in range(epochs):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = sigma_start + (sigma_end - sigma_start) * (epoch / max(1, epochs - 1))
        for b_idx in range(0, len(my_items), micro_batch):
            chunk = my_items[b_idx : b_idx + micro_batch]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                        batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                        batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            accum += 1
            if accum % grad_accum == 0 or (b_idx + micro_batch) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            if rank == 0 and n_batches % 50 == 0:
                print(f"epoch {epoch + 1}/{epochs} step {n_batches} loss {loss.item() * grad_accum:.4f} "
                      f"reward {r.mean().item():.3f}", flush=True)
        if rank == 0:
            print(f"=== epoch {epoch + 1} done in {time.time() - t0:.0f}s, "
                  f"avg loss {epoch_loss / max(1, n_batches):.4f}", flush=True)

    dist.barrier()
    if rank == 0:
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        # save the trained weights first: calibration below must never be able to cost the training run
        os.makedirs(output_dir, exist_ok=True)
        save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
                  os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
        write_config(cfg, output_dir, [1.2, 1.2, 1.2], {})
        print(f"checkpoint saved to {output_dir} with PLACEHOLDER calibration; calibrating next", flush=True)
        try:
            dev_items = torch.load(dev_path, weights_only=False)
            stride = max(1, -(-len(dev_items) // DEV_CALIBRATION_CAP))  # ceil, so at most the cap is kept
            temps, by_bucket = calibrate(model, dev_items[::stride], tok.pad_token_id, device)
        except Exception:
            print(f"CALIBRATION FAILED. Weights and a placeholder rl_agent_config.json are already saved in "
                  f"{output_dir}; calibration can be redone later.", flush=True)
            raise
        write_config(cfg, output_dir, temps, by_bucket)
        print("calibration temperatures (choice, score, noul):", [round(t, 3) for t in temps], flush=True)
        print("calibration temperatures by bucket:", {b: round(t, 3) for b, t in by_bucket.items()}, flush=True)
        print(f"saved to {output_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
