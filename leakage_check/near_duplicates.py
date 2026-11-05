"""Near-duplicate / overlap leakage check between SOS test and train tiles (feeds C1).

Stage 1 (copies): whole-tile normalised correlation at 32x32 (area-downsampled to suppress
  speckle and added noise), max over the 8 dihedral transforms (rotations/flips).
Stage 2 (shifted crops): the central half of each test tile (32x32 at 64x64 scale, i.e. a
  128x128-px region) is template-matched (TM_CCOEFF_NORMED) inside every train tile under all
  8 transforms; the best score says whether that region appears anywhere in training.
Null: Sentinel test tiles vs PALSAR train tiles (different sea and sensor => cannot share
  content). Thresholds = 99.9th percentile of the null maxima.
Also: train-vs-train stage-1 links, then connected components = pseudo-scene clusters.

usage: python near_duplicates.py            (writes to leakage_check/)
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "common"))
sys.path.insert(0, HERE)
from sos_data import list_split  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
cv2.setNumThreads(4)


def d4(a):
    """8 dihedral variants of a 2D array (works on [..., H, W])."""
    v = []
    for k in range(4):
        r = np.rot90(a, k, axes=(-2, -1))
        v += [r, r[..., ::-1]]
    return [np.ascontiguousarray(x) for x in v]


def load(items):
    g64, z32 = [], []
    for _, sp, _ in items:
        im = cv2.imread(sp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        a = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        b = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).ravel()
        b = (b - b.mean()) / (b.std() + 1e-6) / np.sqrt(b.size)
        g64.append(a); z32.append(b)
    return np.stack(g64), np.stack(z32).astype(np.float32)


def stage1(z_q, z_ref, same=False):
    """max correlation of each query vs refs over D4. returns best corr, idx."""
    zq = z_q.reshape(-1, 32, 32)
    best = np.full(len(z_q), -2.0, np.float32); idx = np.zeros(len(z_q), int)
    for v in d4(zq):
        c = v.reshape(len(z_q), -1) @ z_ref.T
        if same:
            np.fill_diagonal(c, -2.0)
        j = c.argmax(1); m = c[np.arange(len(c)), j]
        upd = m > best; best[upd] = m[upd]; idx[upd] = j[upd]
    return best, idx


def mosaic(g64):
    n = len(g64); cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    M = np.zeros((rows * 64, cols * 64), np.float32)
    for k, t in enumerate(g64):
        r, c = divmod(k, cols); M[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = t
    # valid template positions: template (32x32) fully inside one tile -> offsets 0..32
    off = np.arange(33)
    R = (np.arange(n) // cols)[:, None, None] * 64 + off[None, :, None]
    C = (np.arange(n) % cols)[:, None, None] * 64 + off[None, None, :]
    return M, R, C


def stage2(g64_q, ref_mosaic):
    M, R, C = ref_mosaic
    best = np.full(len(g64_q), -2.0, np.float32); idx = np.zeros(len(g64_q), int)
    for i, t in enumerate(g64_q):
        tpl = t[16:48, 16:48]
        for v in d4(tpl):
            res = cv2.matchTemplate(M, v, cv2.TM_CCOEFF_NORMED)
            per_tile = res[R, C].reshape(len(R), -1).max(1)
            j = int(per_tile.argmax())
            if per_tile[j] > best[i]:
                best[i], idx[i] = per_tile[j], j
    return best, idx


def stage2_gpu(g64_q, g64_ref, batch=2):
    """Same as stage2 (normalised cross-correlation of the central 32x32 template inside each
    64x64 reference tile, 8 transforms) computed as batched convolutions on the GPU."""
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda")
    R = torch.from_numpy(g64_ref).float().to(dev)[:, None]                  # N,1,64,64
    s1 = F.avg_pool2d(R, 32, stride=1) * 1024
    s2 = F.avg_pool2d(R * R, 32, stride=1) * 1024
    den = torch.sqrt((s2 - s1 * s1 / 1024).clamp_min(1e-6))                # N,1,33,33
    best = np.full(len(g64_q), -2.0, np.float32); idx = np.zeros(len(g64_q), int)
    with torch.no_grad():
        for s in range(0, len(g64_q), batch):
            tpls = []
            for t in g64_q[s:s + batch]:
                for v in d4(t[16:48, 16:48]):
                    v = v - v.mean(); tpls.append(v / (np.linalg.norm(v) + 1e-6))
            W = torch.from_numpy(np.stack(tpls)).float().to(dev)[:, None]   # 8b,1,32,32
            ncc = F.conv2d(R, W) / den                                      # N,8b,33,33
            m = ncc.amax(dim=(2, 3)).view(len(R), -1, 8).amax(2)           # N,b
            v, j = m.max(0)
            best[s:s + batch] = v.cpu().numpy(); idx[s:s + batch] = j.cpu().numpy()
            del ncc
    return best, idx


def components(n, edges):
    parent = list(range(n))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([f(i) for i in range(n)])


def main():
    t0 = time.time()
    data = {}
    for s in ("sentinel", "palsar"):
        tr, te = list_split("train", s), list_split("test", s)
        data[s] = dict(tr=tr, te=te, g_tr=None, z_tr=None)
        data[s]["g_tr"], data[s]["z_tr"] = load(tr)
        data[s]["g_te"], data[s]["z_te"] = load(te)
        print(f"loaded {s}: {len(tr)} train, {len(te)} test  ({time.time()-t0:.0f}s)", flush=True)

    # (CPU mosaic version kept in stage2() for reference)
    # ---- null distribution: sentinel test vs palsar train ----
    rng = np.random.default_rng(0)
    sub = rng.choice(len(data["sentinel"]["te"]), 300, replace=False)
    n1, _ = stage1(data["sentinel"]["z_te"], data["palsar"]["z_tr"])
    n2, _ = stage2_gpu(data["sentinel"]["g_te"][sub], data["palsar"]["g_tr"])
    thr1, thr2 = float(np.percentile(n1, 99.9)), float(np.percentile(n2, 99.9))
    print(f"null thresholds: copy corr {thr1:.3f}, region match {thr2:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    summary = {"null": {"stage1_p999": thr1, "stage1_max": float(n1.max()),
                        "stage2_p999": thr2, "stage2_max": float(n2.max()), "stage2_null_n": len(sub)}}
    for s in ("sentinel", "palsar"):
        D = data[s]
        b1, i1 = stage1(D["z_te"], D["z_tr"])
        b2, i2 = stage2_gpu(D["g_te"], D["g_tr"])
        print(f"{s}: stages done ({time.time()-t0:.0f}s)", flush=True)
        tt, ti = stage1(D["z_tr"], D["z_tr"], same=True)
        ntr, nte = len(D["tr"]), len(D["te"])
        with open(os.path.join(OUT, f"test_vs_train_{s}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test_id", "copy_corr", "copy_train_id", "region_score", "region_train_id",
                        "is_copy", "region_in_train"])
            for k, (tid, _, _) in enumerate(D["te"]):
                w.writerow([tid, f"{b1[k]:.4f}", D["tr"][i1[k]][0], f"{b2[k]:.4f}", D["tr"][i2[k]][0],
                            int(b1[k] > thr1), int(b2[k] > thr2)])
        # clusters over train+test using all above-threshold links
        edges = [(int(i1[k]), ntr + k) for k in range(nte) if b1[k] > thr1]
        edges += [(int(i2[k]), ntr + k) for k in range(nte) if b2[k] > thr2]
        edges += [(k, int(ti[k])) for k in range(ntr) if tt[k] > thr1]
        comp = components(ntr + nte, edges)
        sizes = np.bincount(np.unique(comp, return_inverse=True)[1])
        test_comps = set(comp[ntr:]); train_comps = set(comp[:ntr])
        summary[s] = {
            "n_test": nte, "n_train": ntr,
            "test_with_copy_in_train": int((b1 > thr1).sum()),
            "test_with_region_in_train": int((b2 > thr2).sum()),
            "test_leaked_either": int(((b1 > thr1) | (b2 > thr2)).sum()),
            "pct_test_leaked_either": float(((b1 > thr1) | (b2 > thr2)).mean() * 100),
            "train_tiles_with_copy_in_train": int((tt > thr1).sum()),
            "copy_corr_percentiles_test": {p: float(np.percentile(b1, p)) for p in (5, 25, 50, 75, 95)},
            "region_score_percentiles_test": {p: float(np.percentile(b2, p)) for p in (5, 25, 50, 75, 95)},
            "clusters_total": int(len(sizes)), "clusters_size_ge_2": int((sizes >= 2).sum()),
            "largest_clusters": sorted(sizes.tolist(), reverse=True)[:10],
            "test_tiles_in_clusters_shared_with_train": int(sum(c in train_comps for c in comp[ntr:])),
        }
        np.savez(os.path.join(OUT, f"raw_{s}.npz"), b1=b1, i1=i1, b2=b2, i2=i2, tt=tt, ti=ti, comp=comp)
        print(s, json.dumps(summary[s], indent=1), flush=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
