import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# ---- PyTorch 2.6 compatibility patch for legacy checkpoints ----
_torch_load_orig = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)

torch.load = _torch_load_compat
# ---------------------------------------------------------------

# Existing local dependency used by the user's current pipeline.
try:
    from pgd import PGD
except Exception as e:  # pragma: no cover - dependency is expected in user's env
    PGD = None
    _pgd_import_error = e
else:
    _pgd_import_error = None


OPENFACE_8_AUS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
AFFWILD2_12_AUS = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
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


@dataclass
class CaseRow:
    case_id: str
    setting: str
    target: str
    source: Optional[str]
    state: Optional[str]
    video_id: str
    frame_id: int
    image_path: str
    baseline_conf: Optional[float] = None
    final_conf: Optional[float] = None
    gain: Optional[float] = None


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_case_table(case_list_path: str, sheet_name: Optional[str] = None) -> List[CaseRow]:
    ext = os.path.splitext(case_list_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(case_list_path)
        chosen = sheet_name or ("All Cases" if "All Cases" in excel.sheet_names else excel.sheet_names[0])
        df = pd.read_excel(case_list_path, sheet_name=chosen)
    else:
        df = pd.read_csv(case_list_path)

    required = ["case_id", "setting", "target", "video_id", "frame_id", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Case list is missing required columns: {missing}")

    rows: List[CaseRow] = []
    for _, row in df.iterrows():
        case_id = str(row["case_id"]).strip()
        setting = normalize_setting(str(row["setting"]).strip())
        target = str(row["target"]).strip()
        source = none_if_blank(row.get("source"))
        state = none_if_blank(row.get("state"))
        video_id = str(row["video_id"]).strip()
        frame_id = int(row["frame_id"])
        image_path = str(row["image_path"]).strip()
        baseline_conf = to_optional_float(row.get("baseline_conf"))
        final_conf = to_optional_float(row.get("final_conf"))
        gain = to_optional_float(row.get("gain"))
        rows.append(CaseRow(case_id, setting, target, source, state, video_id, frame_id, image_path,
                            baseline_conf, final_conf, gain))
    return rows


def none_if_blank(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None


def to_optional_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


def normalize_setting(s: str) -> str:
    x = s.strip().lower().replace("_", "-")
    mapping = {
        "cross-dataset": "cross-dataset",
        "crossdataset": "cross-dataset",
        "cross-model": "cross-model",
        "crossmodel": "cross-model",
        "transfer": "transfer-dataset",
        "transfer-dataset": "transfer-dataset",
        "transferdataset": "transfer-dataset",
    }
    if x not in mapping:
        raise ValueError(f"Unsupported setting '{s}'.")
    return mapping[x]


def resolve_image_path(case: CaseRow, default_root: Optional[str], roots: Dict[str, Optional[str]]) -> str:
    raw = case.image_path
    if os.path.isabs(raw):
        return raw
    candidates = []
    setting_root = roots.get(case.setting)
    if setting_root:
        candidates.append(os.path.join(setting_root, raw))
    if default_root:
        candidates.append(os.path.join(default_root, raw))
    candidates.append(raw)
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_rgb_tensor(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(pil)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
    ])
    tensor = transform(pil).unsqueeze(0)
    return tensor, orig_np


def save_tensor_rgb_image(x: torch.Tensor, out_path: str):
    arr = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def make_delta_visualization(orig: torch.Tensor, adv: torch.Tensor, eps: float) -> Image.Image:
    delta = adv.detach().cpu() - orig.detach().cpu()
    if eps <= 0:
        raise ValueError(f"eps must be positive for perturbation visualization, got {eps}")
    vis = (delta / (2.0 * float(eps))) + 0.5
    vis = torch.clamp(vis, 0.0, 1.0)
    arr = (vis.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_openface_model(model_path: str, device: torch.device):
    from openface.multitask_model import MultitaskPredictor
    predictor = MultitaskPredictor(model_path=model_path, device=device)
    predictor.model.eval()
    return predictor.model


class OpenFaceSingleCaseRunner:
    def __init__(self, model_path: str, device: str, image_size: int, eps: float, alpha: float, steps: int):
        if PGD is None:
            raise ImportError(f"Failed to import PGD from existing codebase: {_pgd_import_error}")
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = load_openface_model(model_path, self.device)
        self.model.eval()
        self.image_size = image_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.attack = PGD(self.model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        self.attack.set_mode_targeted_by_label()

    @torch.no_grad()
    def score_full_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(normalize_imagenet(x.to(self.device)))[2].detach().float().cpu()

    def run(self, image_path: str, target_au: str, source_au: Optional[str], conf_threshold: float) -> Dict:
        x, _ = load_rgb_tensor(image_path, self.image_size)
        if target_au not in OPENFACE_8_AUS:
            raise ValueError(f"Target {target_au} is not available in OpenFace 8 AUs.")
        target_idx = OPENFACE_8_AUS.index(target_au)
        y_t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        s_clean = self.score_full_logits(x)
        x_adv = self.attack(x.to(self.device), y_t).detach().cpu()
        s_adv = self.score_full_logits(x_adv)
        return build_result_dict(
            setting="openface_pgd",
            orig=x,
            adv=x_adv,
            baseline_logits=s_clean,
            final_logits=s_adv,
            target_au=target_au,
            source_au=source_au,
            labels=OPENFACE_8_AUS,
            conf_threshold=conf_threshold,
        )


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
        model.load_state_dict(new_state, strict=False)
        return model

    def _preprocess(self, x_batch: torch.Tensor) -> torch.Tensor:
        x = x_batch.to(self.device).clamp(0.0, 1.0)
        if self.apply_official_resize:
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x = x[:, :, 16:240, 16:240]
        x = normalize_imagenet(x)
        return x

    def score_shared_logits(self, x_batch: torch.Tensor, detach_cpu: bool = False) -> torch.Tensor:
        pred = self.net(self._preprocess(x_batch))
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.float()
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        pred = pred[:, self.shared_indices]
        if detach_cpu:
            return pred.detach().cpu()
        return pred


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
        logits = og_model.score_shared_logits(adv.to(og_model.device), detach_cpu=False)
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

    return adv.detach().cpu()


class OpenGraphAUSelfRunner:
    def __init__(self, repo_path: str, checkpoint_path: str, device: str, image_size: int,
                 eps: float, alpha: float, steps: int, stage: int, arc: str,
                 neighbor_num: int, metric: str, apply_official_resize: bool,
                 lambda_src: float):
        self.image_size = image_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lambda_src = lambda_src
        self.model = OpenGraphAUAttackModel(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            device=device,
            shared_aus=MAIN_SHARED_7_AUS,
            arc=arc,
            stage=stage,
            neighbor_num=neighbor_num,
            metric=metric,
            apply_official_resize=apply_official_resize,
        )

    def run(self, image_path: str, target_au: str, source_au: Optional[str], conf_threshold: float) -> Dict:
        x, _ = load_rgb_tensor(image_path, self.image_size)
        if target_au not in MAIN_SHARED_7_AUS:
            raise ValueError(f"Target {target_au} is not in shared OpenGraphAU set.")
        target_idx = MAIN_SHARED_7_AUS.index(target_au)
        source_idx = MAIN_SHARED_7_AUS.index(source_au) if (source_au in MAIN_SHARED_7_AUS) else None
        s_clean = self.model.score_shared_logits(x, detach_cpu=True)
        x_adv = generate_adv_on_opengraphau(
            og_model=self.model,
            x=x,
            target_index=target_idx,
            source_index=source_idx,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            lambda_src=self.lambda_src,
            random_start=True,
        )
        s_adv = self.model.score_shared_logits(x_adv, detach_cpu=True)
        return build_result_dict(
            setting="opengraphau_self",
            orig=x,
            adv=x_adv,
            baseline_logits=s_clean,
            final_logits=s_adv,
            target_au=target_au,
            source_au=source_au,
            labels=MAIN_SHARED_7_AUS,
            conf_threshold=conf_threshold,
        )


class TransferRunner:
    def __init__(self, repo_path: str, checkpoint_path: str, openface_model_path: str,
                 attack_device: str, eval_device: str, image_size: int,
                 eps: float, alpha: float, steps: int, stage: int, arc: str,
                 neighbor_num: int, metric: str, apply_official_resize: bool,
                 lambda_src: float):
        self.image_size = image_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lambda_src = lambda_src
        self.attack_model = OpenGraphAUAttackModel(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            device=attack_device,
            shared_aus=MAIN_SHARED_7_AUS,
            arc=arc,
            stage=stage,
            neighbor_num=neighbor_num,
            metric=metric,
            apply_official_resize=apply_official_resize,
        )
        self.eval_model = load_openface_model(openface_model_path, torch.device(eval_device if (eval_device != 'cuda' or torch.cuda.is_available()) else 'cpu'))
        self.eval_device = torch.device(eval_device if (eval_device != 'cuda' or torch.cuda.is_available()) else 'cpu')
        self.shared_indices = [OPENFACE_8_AUS.index(au) for au in MAIN_SHARED_7_AUS]

    @torch.no_grad()
    def score_openface_shared(self, x_batch: torch.Tensor) -> torch.Tensor:
        logits = self.eval_model(normalize_imagenet(x_batch.to(self.eval_device)))[2]
        return logits[:, self.shared_indices].detach().cpu()

    def run(self, image_path: str, target_au: str, source_au: Optional[str], conf_threshold: float) -> Dict:
        x, _ = load_rgb_tensor(image_path, self.image_size)
        if target_au not in MAIN_SHARED_7_AUS:
            raise ValueError(f"Target {target_au} is not in shared AU set.")
        target_idx = MAIN_SHARED_7_AUS.index(target_au)
        source_idx = MAIN_SHARED_7_AUS.index(source_au) if (source_au in MAIN_SHARED_7_AUS) else None
        s_clean = self.score_openface_shared(x)
        x_adv = generate_adv_on_opengraphau(
            og_model=self.attack_model,
            x=x,
            target_index=target_idx,
            source_index=source_idx,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            lambda_src=self.lambda_src,
            random_start=True,
        )
        s_adv = self.score_openface_shared(x_adv)
        return build_result_dict(
            setting="transfer_opengraphau_to_openface",
            orig=x,
            adv=x_adv,
            baseline_logits=s_clean,
            final_logits=s_adv,
            target_au=target_au,
            source_au=source_au,
            labels=MAIN_SHARED_7_AUS,
            conf_threshold=conf_threshold,
        )


def build_result_dict(setting: str, orig: torch.Tensor, adv: torch.Tensor,
                      baseline_logits: torch.Tensor, final_logits: torch.Tensor,
                      target_au: str, source_au: Optional[str], labels: List[str],
                      conf_threshold: float) -> Dict:
    b = baseline_logits.squeeze(0).numpy()
    a = final_logits.squeeze(0).numpy()
    target_idx = labels.index(target_au)
    source_idx = labels.index(source_au) if (source_au in labels) else None
    b_top = int(np.argmax(b))
    a_top = int(np.argmax(a))
    baseline_target_score = float(b[target_idx])
    final_target_score = float(a[target_idx])
    baseline_source_score = float(b[source_idx]) if source_idx is not None else None
    final_source_score = float(a[source_idx]) if source_idx is not None else None
    delta = adv - orig
    l_inf = float(delta.abs().max().item())
    l2 = float(torch.norm(delta.view(-1), p=2).item())
    return {
        "setting_runtime": setting,
        "labels": labels,
        "target_au": target_au,
        "source_au": source_au,
        "baseline_top1_label": labels[b_top],
        "baseline_top1_score": float(b[b_top]),
        "final_top1_label": labels[a_top],
        "final_top1_score": float(a[a_top]),
        "baseline_target_score": baseline_target_score,
        "final_target_score": final_target_score,
        "target_gain": final_target_score - baseline_target_score,
        "baseline_target_confident": int((b_top == target_idx) and (baseline_target_score > conf_threshold)),
        "final_target_confident": int((a_top == target_idx) and (final_target_score > conf_threshold)),
        "baseline_target_pass": int(baseline_target_score > 0.5),
        "final_target_pass": int(final_target_score > 0.5),
        "baseline_source_score": baseline_source_score,
        "final_source_score": final_source_score,
        "delta_linf": l_inf,
        "delta_l2": l2,
        "orig_tensor_shape": list(orig.shape),
        "adv_tensor_shape": list(adv.shape),
        "orig": orig,
        "adv": adv,
    }


def save_case_outputs(case: CaseRow, image_path: str, out_dir: str, result: Dict, extra_metadata: Dict, eps: float):
    os.makedirs(out_dir, exist_ok=True)
    orig = result.pop("orig")
    adv = result.pop("adv")

    original_path = os.path.join(out_dir, "original.png")
    perturbed_path = os.path.join(out_dir, "perturbed_true.png")
    perturb_vis_path = os.path.join(out_dir, "perturbation_vis.png")
    metadata_path = os.path.join(out_dir, "metadata.json")

    save_tensor_rgb_image(orig, original_path)
    save_tensor_rgb_image(adv, perturbed_path)
    make_delta_visualization(orig, adv, eps=eps).save(perturb_vis_path)

    metadata = {
        "case_id": case.case_id,
        "setting": case.setting,
        "target": case.target,
        "source": case.source,
        "state": case.state,
        "video_id": case.video_id,
        "frame_id": case.frame_id,
        "image_path_input": case.image_path,
        "perturbation_visualization": {"mode": "eps_normalized", "eps": float(eps), "formula": "vis = clamp(delta / (2*eps) + 0.5, 0, 1)"},
        "image_path_resolved": image_path,
        "artifacts": {
            "original": os.path.basename(original_path),
            "perturbed_true": os.path.basename(perturbed_path),
            "perturbation_vis": os.path.basename(perturb_vis_path),
        },
        **extra_metadata,
        **result,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export figure-ready selected adversarial examples.")
    p.add_argument("--case_list", required=True, help="CSV/XLSX with selected cases.")
    p.add_argument("--sheet_name", default=None, help="Sheet name for XLSX case list. Defaults to 'All Cases' if available.")
    p.add_argument("--export_root", required=True, help="Output directory for exported figures.")

    p.add_argument("--data_root", default=None, help="Fallback root prepended to relative image paths.")
    p.add_argument("--crossmodel_root", default=None)
    p.add_argument("--crossdataset_root", default=None)
    p.add_argument("--transfer_root", default=None)

    p.add_argument("--openface_model", required=True, help="Path to existing OpenFace multitask model.")
    p.add_argument("--og_repo", required=True, help="Path to existing OpenGraphAU repo.")
    p.add_argument("--og_checkpoint", required=True, help="Path to existing OpenGraphAU checkpoint.")

    p.add_argument("--setting", action="append", default=None,
                   help="Optional setting filter. Repeatable: cross-model / cross-dataset / transfer-dataset")
    p.add_argument("--case_id", action="append", default=None, help="Optional case_id filter. Repeatable.")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--eps", type=float, default=8/255)
    p.add_argument("--alpha", type=float, default=3/255)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--conf_threshold", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attack_device", default="cuda")
    p.add_argument("--eval_device", default="cuda")

    p.add_argument("--og_stage", type=int, default=2, choices=[1, 2])
    p.add_argument("--og_arc", default="resnet50")
    p.add_argument("--og_neighbor_num", type=int, default=4)
    p.add_argument("--og_metric", default="dots")
    p.add_argument("--og_apply_official_resize", action="store_true")
    p.add_argument("--og_lambda_src", type=float, default=0.0)
    return p.parse_args()


def maybe_filter_cases(cases: List[CaseRow], settings: Optional[List[str]], case_ids: Optional[List[str]]) -> List[CaseRow]:
    out = cases
    if settings:
        settings_norm = {normalize_setting(s) for s in settings}
        out = [c for c in out if c.setting in settings_norm]
    if case_ids:
        keep = set(case_ids)
        out = [c for c in out if c.case_id in keep]
    return out


def main():
    args = parse_args()
    cases = load_case_table(args.case_list, args.sheet_name)
    cases = maybe_filter_cases(cases, args.setting, args.case_id)
    if not cases:
        raise ValueError("No cases remain after filtering.")

    roots = {
        "cross-model": args.crossmodel_root,
        "cross-dataset": args.crossdataset_root,
        "transfer-dataset": args.transfer_root,
    }

    # Lazy init so users can run only one setting without all deps/devices configured.
    openface_runner = None
    self_runner = None
    transfer_runner = None

    manifest: List[Dict] = []
    for case in cases:
        image_path = resolve_image_path(case, args.data_root, roots)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for {case.case_id}: {image_path}")

        if case.setting == "cross-dataset":
            if openface_runner is None:
                openface_runner = OpenFaceSingleCaseRunner(
                    model_path=args.openface_model,
                    device=args.device,
                    image_size=args.image_size,
                    eps=args.eps,
                    alpha=args.alpha,
                    steps=args.steps,
                )
            result = openface_runner.run(image_path, case.target, case.source, args.conf_threshold)
            extra = {
                "pipeline": "cross-dataset (OpenFace PGD single-case export)",
                "attack_params": {"eps": args.eps, "alpha": args.alpha, "steps": args.steps},
            }
        elif case.setting == "cross-model":
            if self_runner is None:
                self_runner = OpenGraphAUSelfRunner(
                    repo_path=args.og_repo,
                    checkpoint_path=args.og_checkpoint,
                    device=args.attack_device,
                    image_size=args.image_size,
                    eps=args.eps,
                    alpha=args.alpha,
                    steps=args.steps,
                    stage=args.og_stage,
                    arc=args.og_arc,
                    neighbor_num=args.og_neighbor_num,
                    metric=args.og_metric,
                    apply_official_resize=args.og_apply_official_resize,
                    lambda_src=args.og_lambda_src,
                )
            result = self_runner.run(image_path, case.target, case.source, args.conf_threshold)
            extra = {
                "pipeline": "cross-model (OpenGraphAU self-attack single-case export)",
                "attack_params": {"eps": args.eps, "alpha": args.alpha, "steps": args.steps, "lambda_src": args.og_lambda_src},
            }
        elif case.setting == "transfer-dataset":
            if transfer_runner is None:
                transfer_runner = TransferRunner(
                    repo_path=args.og_repo,
                    checkpoint_path=args.og_checkpoint,
                    openface_model_path=args.openface_model,
                    attack_device=args.attack_device,
                    eval_device=args.eval_device,
                    image_size=args.image_size,
                    eps=args.eps,
                    alpha=args.alpha,
                    steps=args.steps,
                    stage=args.og_stage,
                    arc=args.og_arc,
                    neighbor_num=args.og_neighbor_num,
                    metric=args.og_metric,
                    apply_official_resize=args.og_apply_official_resize,
                    lambda_src=args.og_lambda_src,
                )
            result = transfer_runner.run(image_path, case.target, case.source, args.conf_threshold)
            extra = {
                "pipeline": "transfer (OpenGraphAU attack -> OpenFace eval single-case export)",
                "attack_params": {"eps": args.eps, "alpha": args.alpha, "steps": args.steps, "lambda_src": args.og_lambda_src},
            }
        else:
            raise ValueError(f"Unsupported setting: {case.setting}")

        out_dir = os.path.join(args.export_root, case.setting, case.case_id)
        metadata = save_case_outputs(case, image_path, out_dir, result, extra)
        manifest.append({
            "case_id": case.case_id,
            "setting": case.setting,
            "target": case.target,
            "source": case.source,
            "state": case.state,
            "video_id": case.video_id,
            "frame_id": case.frame_id,
            "image_path": image_path,
            "baseline_target_score": metadata["baseline_target_score"],
            "final_target_score": metadata["final_target_score"],
            "target_gain": metadata["target_gain"],
            "baseline_top1_label": metadata["baseline_top1_label"],
            "final_top1_label": metadata["final_top1_label"],
            "baseline_target_confident": metadata["baseline_target_confident"],
            "final_target_confident": metadata["final_target_confident"],
            "export_dir": out_dir,
        })
        print(f"[OK] {case.case_id} -> {out_dir}")

    os.makedirs(args.export_root, exist_ok=True)
    manifest_path_csv = os.path.join(args.export_root, "export_manifest.csv")
    manifest_path_json = os.path.join(args.export_root, "export_manifest.json")
    pd.DataFrame(manifest).to_csv(manifest_path_csv, index=False)
    with open(manifest_path_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path_csv}")
    print(f"Saved manifest: {manifest_path_json}")


if __name__ == "__main__":
    main()
