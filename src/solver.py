"""ARC-AGI-2 low-RAM heuristic solver.
Only numpy + stdlib. Fits 4GB RAM and Kaggle CPU 12h offline.
Strategy: for each task, search a library of deterministic transforms;
pick the first that perfectly explains ALL train pairs, apply to test.
attempt_1 = best rule, attempt_2 = backup (identity or 2nd best).
"""
import json
from copy import deepcopy

import numpy as np


def to_arr(g):
    return np.array(g, dtype=int)

def to_list(a):
    return a.tolist()

# ---------- primitive ops (grid -> grid, or with learned params) ----------

def op_identity(a): return a.copy()
def op_rot90(a): return np.rot90(a, 1)
def op_rot180(a): return np.rot90(a, 2)
def op_rot270(a): return np.rot90(a, 3)
def op_fliph(a): return np.fliplr(a)
def op_flipv(a): return np.flipud(a)
def op_transpose(a): return a.T.copy()
def op_rot90_fliph(a): return np.fliplr(np.rot90(a, 1))

OPS_NO_PARAM = [
    ("identity", op_identity),
    ("rot90", op_rot90),
    ("rot180", op_rot180),
    ("rot270", op_rot270),
    ("fliph", op_fliph),
    ("flipv", op_flipv),
    ("transpose", op_transpose),
    ("rot90_fliph", op_rot90_fliph),
]

def infer_color_map(train_pairs):
    """Learn pixel-wise color mapping from train pairs with same shape.
    Returns dict or None if shapes differ or mapping inconsistent."""
    mapping = {}
    for p in train_pairs:
        inp = to_arr(p["input"]); out = to_arr(p["output"])
        if inp.shape != out.shape:
            return None
        for c_in, c_out in zip(inp.ravel(), out.ravel()):
            c_in, c_out = int(c_in), int(c_out)
            if c_in in mapping and mapping[c_in] != c_out:
                # check if context-dependent -> not a pure recolor
                return None
            mapping[c_in] = c_out
    return mapping

def apply_color_map(a, mapping):
    b = a.copy()
    for k, v in mapping.items():
        b[a == k] = v
    return b

def detect_repeat_tile(train_pairs):
    """Detect if output is input tiled (repeat). Returns (rep_h, rep_w, extra_op) or None.
    Handles e.g. 2x2 -> 6x6 (3x3 tiling with possible flip/rotate of tiles)."""
    for p in train_pairs:
        inp = to_arr(p["input"]); out = to_arr(p["output"])
        ih, iw = inp.shape; oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return None
    # infer reps from first pair
    inp0 = to_arr(train_pairs[0]["input"]); out0 = to_arr(train_pairs[0]["output"])
    rh = out0.shape[0] // inp0.shape[0]; rw = out0.shape[1] // inp0.shape[1]
    # verify all pairs share same reps and tiling exactly reproduces output
    for p in train_pairs:
        inp = to_arr(p["input"]); out = to_arr(p["output"])
        if out.shape[0] // inp.shape[0] != rh or out.shape[1] // inp.shape[1] != rw:
            return None
        tiled = np.tile(inp, (rh, rw))
        if not np.array_equal(tiled, out):
            # try mirrored tile pattern (alternate flip) like example 00576224?
            # example: input [[7,9],[4,3]] -> output row pattern alternates [7,9..] and [9,7..]
            # That's tile with horizontal flip on odd tile-rows? check generic: try flip variants
            found = False
            for op in [op_fliph, op_flipv, op_identity]:
                # try tiling the op(inp)?
                if np.array_equal(np.tile(op(inp), (rh, rw)), out):
                    found = True; break
            if not found:
                # try checkerboard flip tiling
                h, w = inp.shape
                canvas = np.zeros_like(out)
                ok = True
                for i in range(rh):
                    for j in range(rw):
                        tile = inp.copy()
                        # infer flip pattern from first pair's output blocks
                        # compare block (i,j) of out0 with inp0 variants
                        block0 = out0[i*h:(i+1)*h, j*w:(j+1)*w]
                        if np.array_equal(block0, inp0): tile = inp
                        elif np.array_equal(block0, np.fliplr(inp0)): tile = np.fliplr(inp)
                        elif np.array_equal(block0, np.flipud(inp0)): tile = np.flipud(inp)
                        elif np.array_equal(block0, np.rot90(inp0, 2)): tile = np.rot90(inp, 2)
                        else: ok = False; break
                        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = tile
                    if not ok: break
                if not (ok and np.array_equal(canvas, out)):
                    return None
    return (rh, rw)

def apply_repeat_tile(a, reps):
    rh, rw = reps
    return np.tile(a, (rh, rw))

def detect_mirror_tile(train_pairs):
    """Detect alternating mirror tiling (like 00576224: blocks alternate original/flipped).
    Returns block-flip pattern inferred from first pair, or None."""
    p0 = train_pairs[0]
    inp0 = to_arr(p0["input"]); out0 = to_arr(p0["output"])
    ih, iw = inp0.shape; oh, ow = out0.shape
    if oh % ih != 0 or ow % iw != 0: return None
    rh, rw = oh // ih, ow // iw
    pattern = []
    for i in range(rh):
        row = []
        for j in range(rw):
            block = out0[i*ih:(i+1)*ih, j*iw:(j+1)*iw]
            if np.array_equal(block, inp0): row.append("id")
            elif np.array_equal(block, np.fliplr(inp0)): row.append("fh")
            elif np.array_equal(block, np.flipud(inp0)): row.append("fv")
            elif np.array_equal(block, np.rot90(inp0, 2)): row.append("r180")
            else: return None
        pattern.append(row)
    # verify pattern works for all train pairs
    for p in train_pairs[1:]:
        inp = to_arr(p["input"]); out = to_arr(p["output"])
        if out.shape != (inp.shape[0]*rh, inp.shape[1]*rw): return None
        for i in range(rh):
            for j in range(rw):
                tile = {"id": inp, "fh": np.fliplr(inp), "fv": np.flipud(inp), "r180": np.rot90(inp, 2)}[pattern[i][j]]
                if not np.array_equal(out[i*inp.shape[0]:(i+1)*inp.shape[0], j*inp.shape[1]:(j+1)*inp.shape[1]], tile):
                    return None
    return pattern

def apply_mirror_tile(a, pattern):
    ih, iw = a.shape; rh, rw = len(pattern), len(pattern[0])
    out = np.zeros((ih*rh, iw*rw), dtype=int)
    mp = {"id": a, "fh": np.fliplr(a), "fv": np.flipud(a), "r180": np.rot90(a, 2)}
    for i in range(rh):
        for j in range(rw):
            out[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = mp[pattern[i][j]]
    return out

def detect_scale_up(train_pairs):
    """Detect pixel-scale-up (each cell -> kxk block). Returns k or None."""
    for p in train_pairs:
        inp = to_arr(p["input"]); out = to_arr(p["output"])
        ih, iw = inp.shape; oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0: return None
        kh, kw = oh // ih, ow // iw
        if kh != kw: return None
        for i in range(ih):
            for j in range(iw):
                if not np.all(out[i*kh:(i+1)*kh, j*kh:(j+1)*kh] == inp[i, j]):
                    return None
    inp0 = to_arr(train_pairs[0]["input"]); out0 = to_arr(train_pairs[0]["output"])
    return out0.shape[0] // inp0.shape[0]

def detect_crop_to_object(train_pairs):
    """Detect output = minimal bounding box of non-zero? Returns bg color or None."""
    for bg in [0]:
        ok = True
        for p in train_pairs:
            inp = to_arr(p["input"]); out = to_arr(p["output"])
            mask = inp != bg
            if not mask.any():
                ok = False; break
            ys, xs = np.where(mask)
            crop = inp[ys.min():ys.max()+1, xs.min():xs.max()+1]
            if not np.array_equal(crop, out):
                ok = False; break
        if ok: return bg
    return None

def detect_extract_largest_object(train_pairs):
    """Very simple: if output smaller than input and equals one connected component.
    Uses 4-connectivity on non-bg. Returns bg or None."""
    from collections import deque
    def components(a, bg=0):
        h, w = a.shape; seen = np.zeros_like(a, bool); comps = []
        for y in range(h):
            for x in range(w):
                if seen[y, x] or a[y, x] == bg: continue
                q = deque([(y, x)]); seen[y, x] = True; cells = []
                while q:
                    cy, cx = q.popleft(); cells.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] != bg:
                            seen[ny, nx] = True; q.append((ny, nx))
                ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                comps.append(a[min(ys):max(ys)+1, min(xs):max(xs)+1])
        return comps
    for bg in [0]:
        ok = True
        for p in train_pairs:
            inp = to_arr(p["input"]); out = to_arr(p["output"])
            comps = components(inp, bg)
            if not any(np.array_equal(c, out) for c in comps):
                ok = False; break
        if ok: return bg
    return None


def infer_rule(train_pairs):
    """Return (name, apply_fn) that fits all train pairs, or (identity)."""
    # 1. exact op (no param)
    for name, op in OPS_NO_PARAM:
        try:
            if all(np.array_equal(op(to_arr(p["input"])), to_arr(p["output"])) for p in train_pairs):
                return name, op
        except Exception:
            continue
    # 2. op + recolor? try recolor after each op: learn mapping on transformed
    for name, op in OPS_NO_PARAM:
        try:
            transformed = [{"input": to_list(op(to_arr(p["input"]))), "output": p["output"]} for p in train_pairs]
            mp = infer_color_map(transformed)
            if mp is not None:
                # verify
                if all(np.array_equal(apply_color_map(op(to_arr(p["input"])), mp), to_arr(p["output"])) for p in train_pairs):
                    return f"{name}+recolor{mp}", (lambda a, _op=op, _mp=mp: apply_color_map(_op(a), _mp))
        except Exception:
            continue
    # 3. pure recolor
    try:
        mp = infer_color_map(train_pairs)
        if mp is not None:
            return f"recolor{mp}", (lambda a, _mp=mp: apply_color_map(a, _mp))
    except Exception:
        pass
    # 4. mirror tile (covers repeat+mirror)
    try:
        pat = detect_mirror_tile(train_pairs)
        if pat is not None:
            return f"mirror_tile{pat}", (lambda a, _p=pat: apply_mirror_tile(a, _p))
    except Exception:
        pass
    # 5. plain repeat tile
    try:
        reps = detect_repeat_tile(train_pairs)
        if reps is not None:
            return f"tile{reps}", (lambda a, _r=reps: apply_repeat_tile(a, _r))
    except Exception:
        pass
    # 6. scale up
    try:
        k = detect_scale_up(train_pairs)
        if k is not None and k > 1:
            return f"scale{k}", (lambda a, _k=k: np.kron(a, np.ones((_k, _k), dtype=int)))
    except Exception:
        pass
    # 7. crop to bbox
    try:
        bg = detect_crop_to_object(train_pairs)
        if bg is not None:
            def _crop(a, _bg=bg):
                m = a != _bg
                ys, xs = np.where(m)
                return a[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
            return "crop_bbox", _crop
    except Exception:
        pass
    return "identity", op_identity


def solve_task(task):
    train_pairs = task["train"]
    tests = task["test"]
    name, fn = infer_rule(train_pairs)
    preds = []
    for t in tests:
        inp = to_arr(t["input"])
        try:
            p1 = to_list(fn(inp))
        except Exception:
            p1 = to_list(inp)
        # attempt_2: different from attempt_1, simple backup
        if name == "identity":
            p2 = to_list(op_rot90(inp))
        else:
            p2 = to_list(inp)
        preds.append({"attempt_1": p1, "attempt_2": p2})
    return preds, name


def solve_all(challenges):
    out = {}
    rules = {}
    for tid, task in challenges.items():
        preds, rule = solve_task(task)
        out[tid] = preds
        rules[tid] = rule
    return out, rules


def score_predictions(challenges, solutions, predictions):
    """ARC scoring: per test output, 1 if either attempt matches exactly."""
    total = 0; correct = 0
    for tid, task in challenges.items():
        sols = solutions[tid]  # list of grids
        preds = predictions[tid]
        for i, sol in enumerate(sols):
            total += 1
            sol_a = to_arr(sol)
            if np.array_equal(to_arr(preds[i]["attempt_1"]), sol_a) or \
               np.array_equal(to_arr(preds[i]["attempt_2"]), sol_a):
                correct += 1
    return correct, total, (correct / total if total else 0.0)
