import os
import re
import csv
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pgd import PGD

AFFWILD2_12_AUS = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
MODEL_8_AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
STATE_LABELS = {"10": "source1_target0", "11": "source1_target1", "01": "source0_target1", "00": "source0_target0"}
MAIN_STATES = ["10", "01"]


def compute_shared_aus() -> List[str]:
    return [au for au in MODEL_8_AUS if au in AFFWILD2_12_AUS]


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def split_header_or_row(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if "," in line:
        return [x.strip() for x in line.split(",") if x.strip() != ""]
    return re.split(r"\s+", line)


def parse_affwild2_txt(txt_path: str) -> Tuple[List[str], List[List[int]]]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Empty AU annotation file: {txt_path}")
    header = split_header_or_row(lines[0])
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        parts = split_header_or_row(line)
        if len(parts) != len(header):
            raise ValueError(f"Column mismatch in {txt_path} line {i}: expected {len(header)}, got {len(parts)}")
        rows.append([int(x) for x in parts])
    return header, rows


def build_index(
    data_root: str,
    split_dir_name: str = "Validation_Set",
    image_dir_name: str = "cropped_aligned_images",
    sample_every: int = 1,
    max_videos: Optional[int] = None,
) -> List[Dict]:
    split_dir = os.path.join(data_root, split_dir_name)
    image_root = os.path.join(data_root, image_dir_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")
    txt_files = sorted([f for f in os.listdir(split_dir) if f.lower().endswith(".txt")])
    if max_videos is not None:
        txt_files = txt_files[:max_videos]

    index = []
    missing_frames = 0
    missing_video_dirs = 0
    for txt_name in txt_files:
        video_id = os.path.splitext(txt_name)[0]
        txt_path = os.path.join(split_dir, txt_name)
        video_dir = os.path.join(image_root, video_id)
        if not os.path.isdir(video_dir):
            missing_video_dirs += 1
            continue
        header, rows = parse_affwild2_txt(txt_path)
        au_pos = {name: i for i, name in enumerate(header)}
        for frame_idx_1b, row in enumerate(rows, start=1):
            if sample_every > 1 and ((frame_idx_1b - 1) % sample_every != 0):
                continue
            img_path = os.path.join(video_dir, f"{frame_idx_1b:05d}.jpg")
            if not os.path.exists(img_path):
                missing_frames += 1
                continue
            labels, valid = {}, {}
            for au in AFFWILD2_12_AUS:
                if au not in au_pos:
                    labels[au] = 0
                    valid[au] = 0
                    continue
                v = int(row[au_pos[au]])
                if v == -1:
                    labels[au] = 0
                    valid[au] = 0
                else:
                    labels[au] = v
                    valid[au] = 1
            index.append({
                "image_path": img_path,
                "video_id": video_id,
                "frame_id": frame_idx_1b,
                "labels": labels,
                "valid": valid,
            })
    print(f"Indexed {len(index)} frames from {len(txt_files)} txt files")
    if missing_video_dirs:
        print(f"Warning: skipped {missing_video_dirs} videos because image folders were missing")
    if missing_frames:
        print(f"Warning: skipped {missing_frames} frames because jpg files were missing")
    return index


class EntryDataset(Dataset):
    def __init__(self, entries: List[Dict], image_size: int = 224):
        self.entries = entries
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        image = Image.open(e["image_path"]).convert("RGB")
        x = self.transform(image)
        return {"image": x, "video_id": e["video_id"], "frame_id": e["frame_id"], "image_path": e["image_path"]}


def load_openface_model(model_path: str, device: torch.device):
    from openface.multitask_model import MultitaskPredictor
    predictor = MultitaskPredictor(model_path=model_path, device=device)
    predictor.model.eval()
    return predictor.model


@torch.no_grad()
def get_scores(model, x: torch.Tensor) -> torch.Tensor:
    return model(normalize_imagenet(x))[2]


@torch.no_grad()
def random_linf_control(x: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.empty_like(x).uniform_(-eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0)


def build_pairs(shared_aus: List[str], source_au_arg: str, target_au_arg: str) -> List[Tuple[str, str]]:
    def resolve(arg: str):
        if arg.lower() == "all":
            return shared_aus
        if arg not in shared_aus:
            raise ValueError(f"AU {arg} is not in shared AUs: {shared_aus}")
        return [arg]

    srcs = resolve(source_au_arg)
    tgts = resolve(target_au_arg)
    pairs = [(s, t) for s in srcs for t in tgts if s != t]
    if not pairs:
        raise ValueError("No valid (source, target) AU pairs remain")
    return pairs


def filter_entries_by_state(index: List[Dict], source_au: str, target_au: str, state: str) -> List[Dict]:
    if state not in STATE_LABELS:
        raise ValueError(f"Unsupported state {state}. Choose from {list(STATE_LABELS)}")
    s_state = int(state[0])
    t_state = int(state[1])
    out = []
    for e in index:
        valid = e["valid"]
        labels = e["labels"]
        if valid.get(source_au, 0) != 1 or valid.get(target_au, 0) != 1:
            continue
        if labels.get(source_au, 0) != s_state:
            continue
        if labels.get(target_au, 0) != t_state:
            continue
        out.append(e)
    return out


def evaluate_subset(
    model,
    loader,
    device: torch.device,
    source_au: str,
    target_au: str,
    target_index: int,
    attack_eps: float,
    attack_alpha: float,
    attack_steps: int,
    conf_thr: float,
):
    attack = PGD(model, eps=attack_eps, alpha=attack_alpha, steps=attack_steps, random_start=True)
    attack.set_mode_targeted_by_label()

    rows = []
    source_index = MODEL_8_AUS.index(source_au)
    for batch in loader:
        x = batch["image"].to(device)
        bsz = x.shape[0]
        y_t = torch.full((bsz,), target_index, device=device, dtype=torch.long)

        with torch.no_grad():
            s_clean = get_scores(model, x)
        x_adv = attack(x, y_t)
        x_rand = random_linf_control(x, attack_eps)
        with torch.no_grad():
            s_adv = get_scores(model, x_adv)
            s_rand = get_scores(model, x_rand)

        c_np = s_clean.detach().cpu().numpy()
        a_np = s_adv.detach().cpu().numpy()
        r_np = s_rand.detach().cpu().numpy()
        for i in range(bsz):
            c, a, r = c_np[i], a_np[i], r_np[i]
            c_top = int(np.argmax(c))
            a_top = int(np.argmax(a))
            r_top = int(np.argmax(r))
            c_t = float(c[target_index])
            a_t = float(a[target_index])
            r_t = float(r[target_index])
            c_s = float(c[source_index])
            a_s = float(a[source_index])
            r_s = float(r[source_index])
            rows.append({
                "video_id": batch["video_id"][i],
                "frame_id": int(batch["frame_id"][i]),
                "image_path": batch["image_path"][i],
                "baseline_target_top1": int(c_top == target_index),
                "adv_target_top1": int(a_top == target_index),
                "rand_target_top1": int(r_top == target_index),
                "baseline_target_conf": int((c_top == target_index) and (c_t > conf_thr)),
                "adv_target_conf": int((a_top == target_index) and (a_t > conf_thr)),
                "rand_target_conf": int((r_top == target_index) and (r_t > conf_thr)),
                "baseline_target_pass": int(c_t > 0.5),
                "adv_target_pass": int(a_t > 0.5),
                "rand_target_pass": int(r_t > 0.5),
                "adv_joint_success": int((a_top == target_index) and (a_t > conf_thr) and (a_top != source_index)),
                "rand_joint_success": int((r_top == target_index) and (r_t > conf_thr) and (r_top != source_index)),
                "adv_target_gain": float(a_t - c_t),
                "adv_source_drop": float(c_s - a_s),
                "rand_target_gain": float(r_t - c_t),
                "rand_source_drop": float(c_s - r_s),
            })

    if not rows:
        return None, []

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "num_frames": len(rows),
        "baseline_target_top1_rate": avg("baseline_target_top1"),
        "adv_target_top1_rate": avg("adv_target_top1"),
        "rand_target_top1_rate": avg("rand_target_top1"),
        "baseline_target_conf_rate": avg("baseline_target_conf"),
        "adv_target_conf_rate": avg("adv_target_conf"),
        "rand_target_conf_rate": avg("rand_target_conf"),
        "baseline_target_pass_rate": avg("baseline_target_pass"),
        "adv_target_pass_rate": avg("adv_target_pass"),
        "rand_target_pass_rate": avg("rand_target_pass"),
        "adv_joint_success_rate": avg("adv_joint_success"),
        "rand_joint_success_rate": avg("rand_joint_success"),
        "adv_target_score_gain": avg("adv_target_gain"),
        "adv_source_score_drop": avg("adv_source_drop"),
        "rand_target_score_gain": avg("rand_target_gain"),
        "rand_source_score_drop": avg("rand_source_drop"),
    }
    return summary, rows


def macro_average_dicts(items: List[Dict], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        vals = [it[k] for it in items if k in it]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def frame_weighted_average_dicts(items: List[Dict], keys: List[str], weight_key: str = "num_frames") -> Dict[str, float]:
    out = {}
    for k in keys:
        subset = [(float(it[k]), float(it.get(weight_key, 0.0))) for it in items if k in it]
        if not subset:
            out[k] = float("nan")
            continue
        vals = np.array([v for v, _ in subset], dtype=np.float64)
        weights = np.array([w for _, w in subset], dtype=np.float64)
        mask = weights > 0
        out[k] = float(np.average(vals[mask], weights=weights[mask])) if np.any(mask) else float("nan")
    return out


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--source_au", type=str, default="all")
    ap.add_argument("--target_au", type=str, default="all")
    ap.add_argument("--state", type=str, default="all", choices=["all", "10", "11", "01", "00"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--eps", type=float, default=8/255)
    ap.add_argument("--alpha", type=float, default=3/255)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--min_eligible_frames", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def summarize_subset(rows: List[Dict], metric_keys: List[str], states: Optional[List[str]] = None, include_main_states: bool = True) -> Dict:
    out: Dict[str, Dict] = {}
    if not rows:
        return out

    out["macro_over_pair_state"] = {
        "num_pair_states": len(rows),
        **macro_average_dicts(rows, ["num_frames"] + metric_keys),
    }
    out["frame_weighted_over_pair_state"] = {
        "num_pair_states": len(rows),
        **frame_weighted_average_dicts(rows, ["num_frames"] + metric_keys),
    }

    state_groups = states if states is not None else sorted({r["state"] for r in rows})
    by_state_macro = {}
    by_state_weighted = {}
    for state in state_groups:
        subset = [r for r in rows if r["state"] == state]
        if not subset:
            continue
        by_state_macro[state] = {
            "state_label": STATE_LABELS[state],
            "num_pair_states": len(subset),
            **macro_average_dicts(subset, ["num_frames"] + metric_keys),
        }
        by_state_weighted[state] = {
            "state_label": STATE_LABELS[state],
            "num_pair_states": len(subset),
            **frame_weighted_average_dicts(subset, ["num_frames"] + metric_keys),
        }
    out["aggregate_by_state_macro"] = by_state_macro
    out["aggregate_by_state_frame_weighted"] = by_state_weighted

    if include_main_states:
        main_subset = [r for r in rows if r["state"] in MAIN_STATES]
        if main_subset:
            out["main_states_only"] = summarize_subset(main_subset, metric_keys, states=MAIN_STATES, include_main_states=False)

    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    shared_aus = compute_shared_aus()
    print(f"Shared AUs: {shared_aus}")
    index = build_index(args.data_root, sample_every=args.sample_every, max_videos=args.max_videos)
    model = load_openface_model(args.model_path, device)
    pairs = build_pairs(shared_aus, args.source_au, args.target_au)
    states = list(STATE_LABELS.keys()) if args.state == "all" else [args.state]
    print(f"Evaluating {len(pairs)} ordered AU pairs across {len(states)} state(s)")

    pair_rows, frame_rows = [], []
    skipped_small = []
    metric_keys = [
        "baseline_target_top1_rate", "adv_target_top1_rate", "rand_target_top1_rate",
        "baseline_target_conf_rate", "adv_target_conf_rate", "rand_target_conf_rate",
        "baseline_target_pass_rate", "adv_target_pass_rate", "rand_target_pass_rate",
        "adv_joint_success_rate", "rand_joint_success_rate",
        "adv_target_score_gain", "adv_source_score_drop", "rand_target_score_gain", "rand_source_score_drop",
    ]

    for source_au, target_au in pairs:
        target_idx = MODEL_8_AUS.index(target_au)
        for state in states:
            entries = filter_entries_by_state(index, source_au, target_au, state)
            label = STATE_LABELS[state]
            if len(entries) == 0:
                print(f"Skipping {source_au} -> {target_au} [{label}]: 0 eligible frames")
                continue
            if len(entries) < args.min_eligible_frames:
                skipped_small.append({
                    "source_au": source_au,
                    "target_au": target_au,
                    "state": state,
                    "state_label": label,
                    "num_frames": len(entries),
                })
                print(f"Skipping {source_au} -> {target_au} [{label}]: only {len(entries)} eligible frames (< {args.min_eligible_frames})")
                continue

            print(f"\n=== Pair {source_au} -> {target_au} [{label}]: {len(entries)} eligible frames ===")
            loader = DataLoader(
                EntryDataset(entries, image_size=args.image_size),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            summary, rows = evaluate_subset(
                model, loader, device, source_au, target_au, target_idx,
                args.eps, args.alpha, args.steps, args.confidence_threshold,
            )
            if summary is None:
                continue
            pair_row = {
                "source_au": source_au,
                "target_au": target_au,
                "state": state,
                "state_label": label,
                **summary,
            }
            pair_rows.append(pair_row)
            for r in rows:
                r.update({"source_au": source_au, "target_au": target_au, "state": state, "state_label": label})
            frame_rows.extend(rows)
            print(
                f"Baseline top1={summary['baseline_target_top1_rate']:.4f} | "
                f"Adv top1={summary['adv_target_top1_rate']:.4f} | "
                f"Rand top1={summary['rand_target_top1_rate']:.4f} | "
                f"Adv conf={summary['adv_target_conf_rate']:.4f} | "
                f"Adv pass@0.50={summary['adv_target_pass_rate']:.4f} | "
                f"Adv joint={summary['adv_joint_success_rate']:.4f}"
            )

    summary_out = {
        "protocol": "demo_aligned_state_split",
        "shared_aus": shared_aus,
        "states": STATE_LABELS,
        "main_states": {s: STATE_LABELS[s] for s in MAIN_STATES},
        "config": {
            "sample_every": args.sample_every,
            "max_videos": args.max_videos,
            "eps": args.eps,
            "alpha": args.alpha,
            "steps": args.steps,
            "confidence_threshold": args.confidence_threshold,
            "min_eligible_frames": args.min_eligible_frames,
        },
        "num_pair_states_evaluated": len(pair_rows),
        "num_pair_states_skipped_small": len(skipped_small),
        "skipped_small_pair_states": skipped_small,
    }
    if pair_rows:
        summary_out.update(summarize_subset(pair_rows, metric_keys, states=states, include_main_states=True))

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    if pair_rows:
        write_csv(os.path.join(args.out_dir, "pair_state_summary.csv"), pair_rows, list(pair_rows[0].keys()))
    if frame_rows:
        write_csv(os.path.join(args.out_dir, "frame_level_outputs.csv"), frame_rows, list(frame_rows[0].keys()))
    if skipped_small:
        write_csv(os.path.join(args.out_dir, "skipped_small_pair_states.csv"), skipped_small, list(skipped_small[0].keys()))

    print("\n===== DONE =====")
    print(f"Saved summary to: {summary_path}")
    if pair_rows:
        agg = summary_out.get("macro_over_pair_state", {})
        print("\n===== AGGREGATE (macro over pair-states) =====")
        print(f"Pair-states: {len(pair_rows)} | Mean frames per pair-state: {agg.get('num_frames', float('nan')):.2f}")
        print(f"Baseline target top1: {agg.get('baseline_target_top1_rate', float('nan')):.4f}")
        print(f"Adv target top1:      {agg.get('adv_target_top1_rate', float('nan')):.4f}")
        print(f"Rand target top1:     {agg.get('rand_target_top1_rate', float('nan')):.4f}")
        print(f"Adv target @conf:     {agg.get('adv_target_conf_rate', float('nan')):.4f}")
        print(f"Adv target pass:      {agg.get('adv_target_pass_rate', float('nan')):.4f}")
        print(f"Adv joint success:    {agg.get('adv_joint_success_rate', float('nan')):.4f}")
        print(f"Adv target gain:      {agg.get('adv_target_score_gain', float('nan')):.4f}")
        print(f"Adv source drop:      {agg.get('adv_source_score_drop', float('nan')):.4f}")

        state_macro = summary_out.get("aggregate_by_state_macro", {})
        if state_macro:
            print("\n===== AGGREGATE BY STATE (macro over pair-states) =====")
            for state in states:
                state_agg = state_macro.get(state)
                if not state_agg:
                    continue
                print(
                    f"{state} [{STATE_LABELS[state]}] | pairs={state_agg['num_pair_states']} | "
                    f"base={state_agg['baseline_target_top1_rate']:.4f} | "
                    f"adv={state_agg['adv_target_top1_rate']:.4f} | "
                    f"rand={state_agg['rand_target_top1_rate']:.4f} | "
                    f"adv@conf={state_agg['adv_target_conf_rate']:.4f}"
                )

        main = summary_out.get("main_states_only", {}).get("aggregate_by_state_macro", {})
        if main:
            print("\n===== MAIN STATES ONLY (10/01) =====")
            for state in MAIN_STATES:
                state_agg = main.get(state)
                if not state_agg:
                    continue
                print(
                    f"{state} [{STATE_LABELS[state]}] | pairs={state_agg['num_pair_states']} | "
                    f"base={state_agg['baseline_target_top1_rate']:.4f} | "
                    f"adv={state_agg['adv_target_top1_rate']:.4f} | "
                    f"rand={state_agg['rand_target_top1_rate']:.4f}"
                )


if __name__ == "__main__":
    main()
