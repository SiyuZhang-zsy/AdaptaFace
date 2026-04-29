import os
import re
import csv
import json
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
# ---- PyTorch 2.6 compatibility patch for legacy checkpoints ----
_torch_load_orig = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)

torch.load = _torch_load_compat
# ---------------------------------------------------------------
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

AFFWILD2_12_AUS = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
OPENFACE_8_AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
MAIN_SHARED_7_AUS = ["AU1", "AU2", "AU4", "AU6", "AU12", "AU25", "AU26"]
OPENGRAPHAU_41_AUS = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11", "AU12",
    "AU13", "AU14", "AU15", "AU16", "AU17", "AU18", "AU19", "AU20", "AU22", "AU23",
    "AU24", "AU25", "AU26", "AU27", "AU32", "AU38", "AU39",
    "AUL1", "AUR1", "AUL2", "AUR2", "AUL4", "AUR4", "AUL6", "AUR6",
    "AUL10", "AUR10", "AUL12", "AUR12", "AUL14", "AUR14",
]
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
STATE_LABELS = {"10": "source1_target0", "11": "source1_target1", "01": "source0_target1", "00": "source0_target0"}
MAIN_STATES = ["10", "01"]


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

    txt_files = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    txt_files = sorted(txt_files)
    if max_videos is not None:
        txt_files = txt_files[:max_videos]

    index = []
    missing_frames = 0
    missing_video_dirs = 0
    for txt_path in txt_files:
        video_id = os.path.splitext(os.path.basename(txt_path))[0]
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
        return {
            "image": x,
            "video_id": e["video_id"],
            "frame_id": e["frame_id"],
            "image_path": e["image_path"],
        }


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


def load_openface_model(model_path: str, device: torch.device):
    from openface.multitask_model import MultitaskPredictor
    predictor = MultitaskPredictor(model_path=model_path, device=device)
    predictor.model.eval()
    return predictor.model


class OpenFaceEvaluator:
    def __init__(self, model_path: str, device: str = "cuda", shared_aus: Optional[List[str]] = None):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.shared_aus = list(shared_aus or MAIN_SHARED_7_AUS)
        self.shared_indices = [OPENFACE_8_AUS.index(au) for au in self.shared_aus]
        self.model = load_openface_model(model_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def score_batch_shared(self, x_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.model(normalize_imagenet(x_batch.to(self.device)))
        logits = outputs[2]
        return logits[:, self.shared_indices].detach().float().cpu()


class OpenGraphAUAttackModel:
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        shared_aus: Optional[List[str]] = None,
        arc: str = "resnet50",
        neighbor_num: int = 4,
        metric: str = "dots",
        apply_official_resize: bool = True,
        stage: int = 2,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.shared_aus = list(shared_aus or MAIN_SHARED_7_AUS)
        self.all_aus = list(OPENGRAPHAU_41_AUS)
        self.apply_official_resize = apply_official_resize
        self.stage = int(stage)
        self.shared_indices = [self.all_aus.index(au) for au in self.shared_aus]

        if not os.path.isdir(self.repo_path):
            raise FileNotFoundError(f"OpenGraphAU repo path not found: {self.repo_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"OpenGraphAU checkpoint not found: {self.checkpoint_path}")

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        if self.stage == 2:
            from model.MEFL import MEFARG
            net = MEFARG(num_main_classes=27, num_sub_classes=14, backbone=arc)
        else:
            from model.ANFL import MEFARG
            net = MEFARG(num_main_classes=27, num_sub_classes=14, backbone=arc, neighbor_num=neighbor_num, metric=metric)

        self.net = self._load_checkpoint(net, self.checkpoint_path).to(self.device)
        self.net.eval()

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state[k] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[OpenGraphAU] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[OpenGraphAU] Unexpected keys: {len(unexpected)}")
        return model

    def _preprocess(self, x_batch: torch.Tensor) -> torch.Tensor:
        x = x_batch.to(self.device).clamp(0.0, 1.0)
        if self.apply_official_resize:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x = x[:, :, 16:240, 16:240]
        x = normalize_imagenet(x)
        return x

    def score_batch_shared_logits(self, x_batch: torch.Tensor) -> torch.Tensor:
        pred = self.net(self._preprocess(x_batch))
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.float()
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        return pred[:, self.shared_indices]


def random_linf_control(x: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.empty_like(x).uniform_(-eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0)


def generate_adv_on_opengraphau(
    og_model: OpenGraphAUAttackModel,
    x: torch.Tensor,
    target_index: int,
    source_index: Optional[int],
    eps: float,
    alpha: float,
    steps: int,
    lambda_src: float = 0.0,
    random_start: bool = True,
) -> torch.Tensor:
    images = x.clone().detach().to(og_model.device)
    adv = images.clone().detach()

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = torch.clamp(adv, min=0.0, max=1.0).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = og_model.score_batch_shared_logits(adv)
        tgt = logits[:, target_index].mean()
        if source_index is not None and lambda_src > 0.0:
            src = logits[:, source_index].mean()
            objective = tgt - lambda_src * src
        else:
            objective = tgt

        grad = torch.autograd.grad(objective, adv, retain_graph=False, create_graph=False)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, min=-eps, max=eps)
        adv = torch.clamp(images + delta, min=0.0, max=1.0).detach()

    return adv


def summarize_rows(rows: List[Dict]) -> Dict[str, float]:
    def avg(key: str) -> float:
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
        "baseline_target_mean_score": avg("baseline_target_score"),
        "adv_target_mean_score": avg("adv_target_score"),
        "rand_target_mean_score": avg("rand_target_score"),
        "baseline_source_mean_score": avg("baseline_source_score"),
        "adv_source_mean_score": avg("adv_source_score"),
        "rand_source_mean_score": avg("rand_source_score"),
    }
    summary["adv_target_mean_gain"] = summary["adv_target_mean_score"] - summary["baseline_target_mean_score"]
    summary["rand_target_mean_gain"] = summary["rand_target_mean_score"] - summary["baseline_target_mean_score"]
    summary["adv_source_mean_drop"] = summary["baseline_source_mean_score"] - summary["adv_source_mean_score"]
    summary["rand_source_mean_drop"] = summary["baseline_source_mean_score"] - summary["rand_source_mean_score"]
    return summary


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def macro_average_dicts(items: List[Dict], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        vals = [it[k] for it in items if k in it]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def summarize_subset(rows: List[Dict], metric_keys: List[str], states: Optional[List[str]] = None) -> Dict:
    out: Dict[str, Dict] = {}
    if not rows:
        return out
    out["macro_over_pair_state"] = {
        "num_pair_states": len(rows),
        **macro_average_dicts(rows, ["num_frames"] + metric_keys),
    }
    state_groups = states if states is not None else sorted({r["state"] for r in rows})
    by_state_macro = {}
    for state in state_groups:
        subset = [r for r in rows if r["state"] == state]
        if not subset:
            continue
        by_state_macro[state] = {
            "state_label": STATE_LABELS[state],
            "num_pair_states": len(subset),
            **macro_average_dicts(subset, ["num_frames"] + metric_keys),
        }
    out["aggregate_by_state_macro"] = by_state_macro
    return out


def evaluate_subset_cross_model(
    og_attack_model: OpenGraphAUAttackModel,
    openface_eval: OpenFaceEvaluator,
    loader,
    shared_aus: List[str],
    source_au: str,
    target_au: str,
    attack_eps: float,
    attack_alpha: float,
    attack_steps: int,
    conf_thr: float,
    pass_thr: float,
    state_code: str = "",
    state_label: str = "",
    lambda_src: float = 0.0,
):
    rows = []
    source_index = shared_aus.index(source_au)
    target_index = shared_aus.index(target_au)

    for batch in loader:
        x = batch["image"]
        x_adv = generate_adv_on_opengraphau(
            og_model=og_attack_model,
            x=x,
            target_index=target_index,
            source_index=source_index,
            eps=attack_eps,
            alpha=attack_alpha,
            steps=attack_steps,
            lambda_src=lambda_src,
            random_start=True,
        )
        x_rand = random_linf_control(x.to(og_attack_model.device), attack_eps)

        s_clean = openface_eval.score_batch_shared(x)
        s_adv = openface_eval.score_batch_shared(x_adv)
        s_rand = openface_eval.score_batch_shared(x_rand)

        c_np = s_clean.detach().cpu().numpy()
        a_np = s_adv.detach().cpu().numpy()
        r_np = s_rand.detach().cpu().numpy()
        bsz = c_np.shape[0]

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
            c_top_score = float(c[c_top])
            a_top_score = float(a[a_top])
            r_top_score = float(r[r_top])

            rows.append({
                "video_id": batch["video_id"][i],
                "frame_id": int(batch["frame_id"][i]),
                "state": state_code,
                "state_label": state_label,
                "image_path": batch["image_path"][i],
                "baseline_top1_au": shared_aus[c_top],
                "adv_top1_au": shared_aus[a_top],
                "rand_top1_au": shared_aus[r_top],
                "baseline_top1_score": c_top_score,
                "adv_top1_score": a_top_score,
                "rand_top1_score": r_top_score,
                "baseline_target_score": c_t,
                "adv_target_score": a_t,
                "rand_target_score": r_t,
                "baseline_source_score": c_s,
                "adv_source_score": a_s,
                "rand_source_score": r_s,
                "baseline_target_top1": int(c_top == target_index),
                "adv_target_top1": int(a_top == target_index),
                "rand_target_top1": int(r_top == target_index),
                "baseline_target_conf": int((c_top == target_index) and (c_t > conf_thr)),
                "adv_target_conf": int((a_top == target_index) and (a_t > conf_thr)),
                "rand_target_conf": int((r_top == target_index) and (r_t > conf_thr)),
                "baseline_target_pass": int(c_t > pass_thr),
                "adv_target_pass": int(a_t > pass_thr),
                "rand_target_pass": int(r_t > pass_thr),
                "adv_joint_success": int((a_top == target_index) and (a_t > conf_thr) and (a_t > a_s)),
                "rand_joint_success": int((r_top == target_index) and (r_t > conf_thr) and (r_t > r_s)),
                "adv_target_gain": float(a_t - c_t),
                "adv_source_drop": float(c_s - a_s),
                "rand_target_gain": float(r_t - c_t),
                "rand_source_drop": float(c_s - r_s),
            })

    if not rows:
        return None, []
    return summarize_rows(rows), rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True, help="Path to OpenFace multitask AU model")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--source_au", type=str, default="all")
    ap.add_argument("--target_au", type=str, default="all")
    ap.add_argument("--state", type=str, default="all", choices=["all", "10", "11", "01", "00"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--alpha", type=float, default=3 / 255)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--pass_threshold", type=float, default=0.5)
    ap.add_argument("--min_eligible_frames", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--og_repo", type=str, required=True)
    ap.add_argument("--og_checkpoint", type=str, required=True)
    ap.add_argument("--og_device", type=str, default="cuda")
    ap.add_argument("--og_arc", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet101", "swin_transformer_tiny", "swin_transformer_small", "swin_transformer_base"])
    ap.add_argument("--og_stage", type=int, default=None, choices=[1, 2])
    ap.add_argument("--og_neighbor_num", type=int, default=4)
    ap.add_argument("--og_metric", type=str, default="dots")
    ap.add_argument("--og_apply_official_resize", action="store_true")
    ap.add_argument("--og_lambda_src", type=float, default=0.0, help="Source suppression weight for OpenGraphAU attack objective")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    shared_aus = list(MAIN_SHARED_7_AUS)
    print(f"Cross-model shared AUs: {shared_aus}")

    index = build_index(args.data_root, sample_every=args.sample_every, max_videos=args.max_videos)

    og_stage = args.og_stage
    if og_stage is None:
        ckpt_name = os.path.basename(args.og_checkpoint).lower()
        og_stage = 2 if ("second_stage" in ckpt_name or "stage2" in ckpt_name) else 1
        print(f"[OpenGraphAU] Inferred stage={og_stage} from checkpoint name: {os.path.basename(args.og_checkpoint)}")

    og_attack_model = OpenGraphAUAttackModel(
        repo_path=args.og_repo,
        checkpoint_path=args.og_checkpoint,
        device=args.og_device,
        shared_aus=shared_aus,
        arc=args.og_arc,
        stage=og_stage,
        neighbor_num=args.og_neighbor_num,
        metric=args.og_metric,
        apply_official_resize=args.og_apply_official_resize,
    )
    openface_eval = OpenFaceEvaluator(args.model_path, device=args.device, shared_aus=shared_aus)

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
        "baseline_target_mean_score", "adv_target_mean_score", "rand_target_mean_score",
        "baseline_source_mean_score", "adv_source_mean_score", "rand_source_mean_score",
        "adv_target_mean_gain", "rand_target_mean_gain", "adv_source_mean_drop", "rand_source_mean_drop",
    ]

    for source_au, target_au in pairs:
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
                pin_memory=(torch.cuda.is_available() and args.device == "cuda"),
            )
            summary, rows = evaluate_subset_cross_model(
                og_attack_model=og_attack_model,
                openface_eval=openface_eval,
                loader=loader,
                shared_aus=shared_aus,
                source_au=source_au,
                target_au=target_au,
                attack_eps=args.eps,
                attack_alpha=args.alpha,
                attack_steps=args.steps,
                conf_thr=args.confidence_threshold,
                pass_thr=args.pass_threshold,
                state_code=state,
                state_label=label,
                lambda_src=args.og_lambda_src,
            )
            if summary is None:
                continue
            pair_row = {"source_au": source_au, "target_au": target_au, "state": state, "state_label": label, **summary}
            pair_rows.append(pair_row)
            for r in rows:
                r.update({"source_au": source_au, "target_au": target_au, "state": state, "state_label": label})
            frame_rows.extend(rows)
            print(
                f"Baseline top1={summary['baseline_target_top1_rate']:.4f} | "
                f"Adv top1={summary['adv_target_top1_rate']:.4f} | "
                f"Rand top1={summary['rand_target_top1_rate']:.4f} | "
                f"Adv conf={summary['adv_target_conf_rate']:.4f} | "
                f"Adv pass@{args.pass_threshold:.2f}={summary['adv_target_pass_rate']:.4f} | "
                f"Adv joint={summary['adv_joint_success_rate']:.4f}"
            )
            print(
                f"Target mean score | Base={summary['baseline_target_mean_score']:.4f} | Adv={summary['adv_target_mean_score']:.4f} | "
                f"Rand={summary['rand_target_mean_score']:.4f} | AdvΔ={summary['adv_target_mean_gain']:.4f}"
            )
            print(
                f"Source mean score | Base={summary['baseline_source_mean_score']:.4f} | Adv={summary['adv_source_mean_score']:.4f} | "
                f"Rand={summary['rand_source_mean_score']:.4f} | AdvDrop={summary['adv_source_mean_drop']:.4f}"
            )

    summary_out = {
        "protocol": "cross_model_opengraphau_to_openface",
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
            "pass_threshold": args.pass_threshold,
            "min_eligible_frames": args.min_eligible_frames,
            "og_arc": args.og_arc,
            "og_repo": args.og_repo,
            "og_checkpoint": args.og_checkpoint,
            "og_stage": og_stage,
            "og_lambda_src": args.og_lambda_src,
        },
        "num_pair_states_evaluated": len(pair_rows),
        "num_pair_states_skipped_small": len(skipped_small),
        "skipped_small_pair_states": skipped_small,
    }
    if pair_rows:
        summary_out.update(summarize_subset(pair_rows, metric_keys, states=states))

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
        print(f"Base target mean:     {agg.get('baseline_target_mean_score', float('nan')):.4f}")
        print(f"Adv target mean:      {agg.get('adv_target_mean_score', float('nan')):.4f}")
        print(f"Rand target mean:     {agg.get('rand_target_mean_score', float('nan')):.4f}")
        print(f"Adv target mean Δ:    {agg.get('adv_target_mean_gain', float('nan')):.4f}")
        print(f"Base source mean:     {agg.get('baseline_source_mean_score', float('nan')):.4f}")
        print(f"Adv source mean:      {agg.get('adv_source_mean_score', float('nan')):.4f}")
        print(f"Adv source mean drop: {agg.get('adv_source_mean_drop', float('nan')):.4f}")

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


if __name__ == "__main__":
    main()
