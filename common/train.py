"""Generic trainer for M2 / M3 (M1 keeps its own faithful script, train_cbdnet.py).

Recipes follow each paper's reported settings (review Section 7); unreported choices are marked
'assumed' in models.py / here.
- M2: no validation set in its paper, so the final-epoch checkpoint is evaluated (saved as best.pth).
- M3: the paper holds out 10 % of the official training set (sampled once, fixed) and evaluates the checkpoint
  with the best validation score. We do the same (val_frac=0.1, best validation mIoU, single pass). With a
  group split the CSV's own 'val' partition is the held-out set and training uses train + calib.
- Mixed precision and gradient accumulation (amp / accum) only fit a large model into 4 GB of GPU memory;
  the effective batch size stays as in the paper.

usage (from experiments/):  python common/train.py --arch dlv3p_mnv2 --sensor sentinel --seed 0
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXP, "third_party", "CBDNet-main"))

from dice_bce_loss import dice_bce_loss       # noqa: E402  (BCE + Dice, equal weights)
from models import build_model, complexity    # noqa: E402
from sos_data import SOSTrain, list_from_split_csv, list_split  # noqa: E402
from evaluate import Acc  # noqa: E402  (pooled metrics, same as the test evaluator)

RECIPES = {
    # [P01]: Adam, lr 1e-3, batch 15, 80 epochs, BCE+Dice. Assumed: cosine decay, flips/rot90.
    "dlv3p_mnv2": dict(lr=1e-3, bs=15, epochs=80, aug="flips", tag="M2"),
    # [P29] Sec. 4.1: Adam, lr 1e-3, batch 16, 80 epochs, BCE+Dice, 256x256, best-validation checkpoint.
    # Assumed: cosine decay, flips/rot90. Batch 16 = 2 x 8 with gradient accumulation, mixed precision.
    "fcsnet": dict(lr=1e-3, bs=8, accum=2, amp=True, val_frac=0.1, epochs=80, aug="flips", tag="M3"),
}


def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(RECIPES))
    ap.add_argument("--sensor", default="sentinel", choices=["sentinel", "palsar"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--workers", type=int, default=2)  # 2 keeps RAM low on a 16 GB laptop
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pth if it exists")
    ap.add_argument("--split", default="", help="group-split CSV (relative to experiments/); empty = official split")
    ap.add_argument("--partitions", default="train,val,calib", help="partitions to train on (with --split)")
    a = ap.parse_args()
    R = dict(RECIPES[a.arch]); R["epochs"] = a.epochs or R["epochs"]; R["bs"] = a.bs or R["bs"]

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    name = (f"{R['tag']}_{a.arch}_{a.sensor}_s{a.seed}" + ("_groupsplit" if a.split else "")
            + (f"_debug{a.limit}" if a.limit else ""))
    out = os.path.join(EXP, "runs", name); os.makedirs(out, exist_ok=True)

    accum, amp, vf = R.get("accum", 1), R.get("amp", False), R.get("val_frac", 0)
    if a.split:
        parts = a.partitions.split(",")
        if vf:  # the CSV's 'val' partition is the held-out set
            parts = [p for p in parts if p != "val"]
            val_items = list_from_split_csv(os.path.join(EXP, a.split), ["val"])
        items = list_from_split_csv(os.path.join(EXP, a.split), parts)
    else:
        items = list_split("train", a.sensor)
        if vf:  # paper: 10 % of the official training images, sampled once with a fixed seed
            perm = np.random.RandomState(1234).permutation(len(items))
            nv = int(round(vf * len(items)))
            val_items = [items[i] for i in sorted(perm[:nv])]
            items = [items[i] for i in sorted(perm[nv:])]
    items = items[: a.limit or None]
    if vf:
        val_items = val_items[: (a.limit // 5 or None) if a.limit else None]
        with open(os.path.join(out, "val_ids.txt"), "w") as f:
            f.write("\n".join(str(i[0]) for i in val_items))
        vdl = torch.utils.data.DataLoader(SOSTrain(val_items, augment=False), batch_size=R["bs"], shuffle=False,
                                          num_workers=a.workers)
    dl = torch.utils.data.DataLoader(SOSTrain(items, aug=R["aug"]), batch_size=R["bs"], shuffle=True,
                                     num_workers=a.workers, pin_memory=a.device == "cuda", drop_last=True,
                                     persistent_workers=a.workers > 0, worker_init_fn=seed_worker)
    dev = torch.device(a.device)
    net = build_model(a.arch).to(dev)
    params, gmacs = complexity(net)
    opt = torch.optim.Adam(net.parameters(), lr=R["lr"])
    total = R["epochs"] * (len(dl) // accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * it / total)))
    crit = dice_bce_loss()
    scaler = torch.amp.GradScaler(enabled=amp)
    print(f"{name}: {len(items)} tiles" + (f" (+{len(val_items)} validation)" if vf else "")
          + f", {len(dl) // accum} updates/epoch, {params:.2f}M params, {gmacs:.2f} GMACs, {R}", flush=True)

    def validate():
        net.eval(); acc = Acc()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            for img, mask in vdl:
                p = net(img.to(dev)).float().cpu().numpy()[:, 0]
                for k in range(len(p)):
                    acc.add(None, p[k], mask[k, 0].numpy() > 0.5)
        return acc.result()["miou"]

    ckpt_path, start, t0, best_val = os.path.join(out, "last.pth"), 1, 0, -1.0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"].cpu())
        start, t0 = ck["epoch"] + 1, ck["seconds"]
        best_val = ck.get("best_val", -1.0); scaler.load_state_dict(ck.get("scaler", scaler.state_dict()))
        print(f"resumed from epoch {ck['epoch']}", flush=True)
    log = open(os.path.join(out, "train_log.csv"), "a" if start > 1 else "w", newline="")
    w = csv.writer(log)
    if start == 1:
        w.writerow(["epoch", "train_loss", "lr", "seconds"] + (["val_miou", "saved"] if vf else []))
    tic = time.time() - t0
    for ep in range(start, R["epochs"] + 1):
        net.train(); tot = 0.0
        opt.zero_grad()
        for it, (img, mask) in enumerate(dl, 1):
            img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                pred = net(img)
            loss = crit(mask, pred.float())          # loss in float32 (BCE is not autocast-safe)
            scaler.scale(loss / accum).backward()
            if it % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            tot += loss.item()
        lr = opt.param_groups[0]["lr"]
        row = [ep, f"{tot/len(dl):.5f}", f"{lr:.2e}", int(time.time() - tic)]
        msg = f"epoch {ep:3d}  loss {tot/len(dl):.4f}  lr {lr:.1e}  {int(time.time()-tic)}s"
        if vf:   # keep the checkpoint with the best validation mIoU (paper protocol)
            v = validate(); saved = v > best_val
            if saved:
                best_val = v; torch.save(net.state_dict(), os.path.join(out, "best.pth"))
            row += [f"{v:.5f}", int(saved)]; msg += f"  val mIoU {v:.4f}" + ("  *" if saved else "")
        else:
            torch.save(net.state_dict(), os.path.join(out, "best.pth"))  # final epoch = evaluated model
        w.writerow(row); log.flush(); print(msg, flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                        best_val=best_val, scaler=scaler.state_dict(),
                        seconds=int(time.time() - tic), py_rng=random.getstate(), np_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state()), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)  # full state, so an interrupted run can --resume
    log.close()
    print("done", name)


if __name__ == "__main__":
    main()
