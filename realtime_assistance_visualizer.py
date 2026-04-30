
import argparse
import json
import time
import tempfile
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor
from pgd import PGD


AU_LIST = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
AU_TO_IDX = {au: i for i, au in enumerate(AU_LIST)}
IDX_TO_AU = {i: au for i, au in enumerate(AU_LIST)}

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_tensor(frame_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=True),
    ])
    return tfm(rgb).unsqueeze(0)


def compute_delta_vis(orig_rgb_tensor: torch.Tensor, adv_rgb_tensor: torch.Tensor, mode: str = "per_image", eps: float = 8 / 255) -> np.ndarray:
    delta = adv_rgb_tensor - orig_rgb_tensor
    if mode == "per_image":
        max_abs = float(delta.abs().max().item())
        denom = 2 * max(max_abs, 1e-8)
    elif mode == "eps":
        denom = 2 * max(float(eps), 1e-8)
    else:
        raise ValueError(f"Unsupported vis mode: {mode}")
    vis = clamp01(delta / denom + 0.5)
    return tensor_to_bgr_uint8(vis)


def topk_pairs(scores: np.ndarray, k: int = 2) -> List[Tuple[str, float]]:
    idxs = np.argsort(scores)[::-1][:k]
    return [(IDX_TO_AU[int(i)], float(scores[int(i)])) for i in idxs]


def put_multiline_text(canvas: np.ndarray, lines: List[str], x: int, y: int, line_gap: int = 24, color: Tuple[int, int, int] = (255, 255, 255), scale: float = 0.6, thickness: int = 1, bg: bool = True):
    for i, line in enumerate(lines):
        yy = y + i * line_gap
        if bg:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.rectangle(canvas, (x - 4, yy - h - 6), (x + w + 6, yy + 4), (0, 0, 0), -1)
        cv2.putText(canvas, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class FaceAttackVisualizer:
    def __init__(self, face_model_path: str, multitask_model_path: str, device: str = "cuda", crop_size: int = 224, eps: float = 8 / 255, alpha: float = 3 / 255, steps: int = 3):
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.crop_size = crop_size
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)

        self.face_detector = FaceDetector(model_path=face_model_path, device=self.device)
        self.multitask_predictor = MultitaskPredictor(model_path=multitask_model_path, device=self.device)
        self.multitask_predictor.model.eval()

    def extract_face_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Compatibility wrapper:
        - some openface builds accept ndarray frames directly
        - some only accept image paths
        """
        try:
            face_result = self.face_detector.get_face(frame_bgr)
        except Exception:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                ok = cv2.imwrite(tmp_path, frame_bgr)
                if not ok:
                    return None
                face_result = self.face_detector.get_face(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        if isinstance(face_result, tuple):
            crop = face_result[0]
        else:
            crop = face_result
        return crop

    def predict_scores_from_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        preds = self.multitask_predictor.predict(crop_bgr)
        _, _, au_output = preds
        return au_output[0].detach().cpu().numpy().astype(np.float32)

    def attack_crop(self, crop_bgr: np.ndarray, target_au: str, eps: Optional[float] = None, alpha: Optional[float] = None, steps: Optional[int] = None):
        if target_au not in AU_TO_IDX:
            raise ValueError(f"Unsupported target AU: {target_au}. Choices: {list(AU_TO_IDX)}")

        eps = self.eps if eps is None else float(eps)
        alpha = self.alpha if alpha is None else float(alpha)
        steps = self.steps if steps is None else int(steps)

        attack = PGD(self.multitask_predictor.model, eps=eps, alpha=alpha, steps=steps, random_start=True)

        x = bgr_to_rgb_tensor(crop_bgr, size=self.crop_size).to(self.device)
        y = torch.tensor([AU_TO_IDX[target_au]], dtype=torch.long, device=self.device)
        x_adv = attack(x, y)

        orig_bgr_224 = tensor_to_bgr_uint8(x)
        adv_bgr_224 = tensor_to_bgr_uint8(x_adv)
        return adv_bgr_224, x.detach().cpu(), x_adv.detach().cpu(), orig_bgr_224

    def process_frame(self, frame_bgr: np.ndarray, target_au: str, vis_mode: str = "per_image") -> Dict:
        crop = self.extract_face_crop(frame_bgr)
        if crop is None:
            return {"has_face": False, "frame_bgr": frame_bgr}

        scores_orig = self.predict_scores_from_crop(crop)
        adv_bgr_224, orig_tensor_224, adv_tensor_224, orig_bgr_224 = self.attack_crop(crop, target_au)
        scores_adv = self.predict_scores_from_crop(adv_bgr_224)

        delta_vis_bgr = compute_delta_vis(orig_tensor_224, adv_tensor_224, mode=vis_mode, eps=self.eps)

        top1_orig_idx = int(np.argmax(scores_orig))
        top1_orig_label = IDX_TO_AU[top1_orig_idx]
        top1_orig_score = float(scores_orig[top1_orig_idx])

        top2_adv = topk_pairs(scores_adv, k=2)
        target_idx = AU_TO_IDX[target_au]

        return {
            "has_face": True,
            "frame_bgr": frame_bgr,
            "orig_crop_disp_bgr": orig_bgr_224,
            "adv_crop_disp_bgr": adv_bgr_224,
            "delta_vis_bgr": delta_vis_bgr,
            "top1_orig_label": top1_orig_label,
            "top1_orig_score": top1_orig_score,
            "target_label": target_au,
            "target_orig_score": float(scores_orig[target_idx]),
            "target_adv_score": float(scores_adv[target_idx]),
            "top2_adv": top2_adv,
        }


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def stack_h(images: List[np.ndarray], pad: int = 8, bg_color: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    max_h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] != max_h:
            im = resize_to_height(im, max_h)
        padded.append(im)
    total_w = sum(im.shape[1] for im in padded) + pad * (len(padded) - 1)
    canvas = np.full((max_h, total_w, 3), bg_color, dtype=np.uint8)
    x = 0
    for im in padded:
        h, w = im.shape[:2]
        canvas[:h, x:x+w] = im
        x += w + pad
    return canvas


def stack_v(images: List[np.ndarray], pad: int = 8, bg_color: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    max_w = max(im.shape[1] for im in images)
    total_h = sum(im.shape[0] for im in images) + pad * (len(images) - 1)
    canvas = np.full((total_h, max_w, 3), bg_color, dtype=np.uint8)
    y = 0
    for im in images:
        h, w = im.shape[:2]
        canvas[y:y+h, :w] = im
        y += h + pad
    return canvas


def make_text_panel(width: int, height: int, lines: List[str], title: Optional[str] = None) -> np.ndarray:
    panel = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    y = 34
    if title:
        put_multiline_text(panel, [title], 12, y, line_gap=28, scale=0.75, thickness=2, color=(255, 255, 255), bg=False)
        y += 34
    put_multiline_text(panel, lines, 12, y, line_gap=26, scale=0.6, thickness=1, color=(230, 230, 230), bg=False)
    return panel


def annotate_image(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose_view(processed: Dict, target_au: str, vis_mode: str) -> np.ndarray:
    full = resize_to_height(processed["frame_bgr"].copy(), 540)

    if not processed["has_face"]:
        put_multiline_text(full, ["No face detected.", f"Target AU: {target_au}"], 16, 32, line_gap=28, scale=0.8, thickness=2, bg=True)
        empty = np.full((540, 720, 3), (20, 20, 20), dtype=np.uint8)
        put_multiline_text(empty, ["No face crop available."], 20, 50, line_gap=28, scale=0.8, thickness=2, bg=False)
        return stack_h([full, empty], pad=12)

    left = full.copy()
    put_multiline_text(left, [f"Original Top-1: {processed['top1_orig_label']}", f"Confidence: {processed['top1_orig_score']:.3f}"], 14, 36, line_gap=28, scale=0.7, thickness=2, bg=True)

    orig_face = annotate_image(processed["orig_crop_disp_bgr"], "Face Crop")
    adv_face = annotate_image(processed["adv_crop_disp_bgr"], "Perturbed Face")
    delta_face = annotate_image(processed["delta_vis_bgr"], f"Perturbation ({vis_mode})")
    image_row = stack_h([orig_face, adv_face, delta_face], pad=8)

    top2 = processed["top2_adv"]
    text_lines = [
        f"Target AU: {processed['target_label']}",
        f"Target score (orig): {processed['target_orig_score']:.3f}",
        f"Target score (adv):  {processed['target_adv_score']:.3f}",
        "",
        f"Adv Top-1: {top2[0][0]} ({top2[0][1]:.3f})",
        f"Adv Top-2: {top2[1][0]} ({top2[1][1]:.3f})",
    ]
    text_panel = make_text_panel(image_row.shape[1], 140, text_lines, title="Adversarial Assistance")
    right = stack_v([image_row, text_panel], pad=8)

    max_h = max(left.shape[0], right.shape[0])
    if left.shape[0] < max_h:
        left = np.vstack([left, np.full((max_h - left.shape[0], left.shape[1], 3), (20, 20, 20), dtype=np.uint8)])
    if right.shape[0] < max_h:
        right = np.vstack([right, np.full((max_h - right.shape[0], right.shape[1], 3), (20, 20, 20), dtype=np.uint8)])

    return stack_h([left, right], pad=12)


def open_video_source(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def open_screen_source():
    try:
        import mss
        return mss.mss()
    except ImportError as e:
        raise RuntimeError("Screen capture requires `mss`. Install it with: pip install mss") from e


def grab_screen_frame(sct, monitor_idx: int = 1, region_json: Optional[str] = None) -> np.ndarray:
    if region_json:
        region = json.loads(region_json)
        monitor = {"left": int(region["left"]), "top": int(region["top"]), "width": int(region["width"]), "height": int(region["height"])}
    else:
        monitor = sct.monitors[int(monitor_idx)]
    shot = sct.grab(monitor)
    frame = np.array(shot)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def parse_args():
    ap = argparse.ArgumentParser(description="Real-time facial adversarial assistance visualizer")
    ap.add_argument("--source", choices=["video", "screen"], required=True)
    ap.add_argument("--video_path", type=str, default=None)
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region_json", type=str, default=None)
    ap.add_argument("--face_model", type=str, default="./weights/Alignment_RetinaFace.pth")
    ap.add_argument("--multitask_model", type=str, default="./weights/MTL_backbone.pth")
    ap.add_argument("--target_au", type=str, required=True, choices=AU_LIST)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--crop_size", type=int, default=224)
    ap.add_argument("--eps", type=float, default=8/255)
    ap.add_argument("--alpha", type=float, default=3/255)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--vis_mode", choices=["per_image", "eps"], default="per_image")
    ap.add_argument("--attack_every", type=int, default=1)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--save_fps", type=float, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    vis = FaceAttackVisualizer(args.face_model, args.multitask_model, device=args.device, crop_size=args.crop_size, eps=args.eps, alpha=args.alpha, steps=args.steps)

    writer = None
    last_processed = None
    frame_idx = 0

    if args.source == "video":
        if not args.video_path:
            raise ValueError("--video_path is required when --source video")
        cap = open_video_source(args.video_path)
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if not input_fps or input_fps <= 0:
            input_fps = 25.0
        output_fps = args.save_fps or input_fps

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(1, args.attack_every) == 0 or last_processed is None:
                last_processed = vis.process_frame(frame, args.target_au, vis_mode=args.vis_mode)
            else:
                last_processed["frame_bgr"] = frame

            canvas = compose_view(last_processed, args.target_au, args.vis_mode)

            if writer is None and args.save_path:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))

            if writer is not None:
                writer.write(canvas)

            if args.display:
                cv2.imshow("Adversarial Assistance Visualizer", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    while True:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 in (ord("p"), ord("q")):
                            if k2 == ord("q"):
                                cap.release()
                                if writer is not None:
                                    writer.release()
                                cv2.destroyAllWindows()
                                return
                            break
            frame_idx += 1
        cap.release()

    else:
        sct = open_screen_source()
        output_fps = args.save_fps or 12.0
        frame_interval = 1.0 / output_fps

        while True:
            t0 = time.time()
            frame = grab_screen_frame(sct, monitor_idx=args.monitor, region_json=args.region_json)

            if frame_idx % max(1, args.attack_every) == 0 or last_processed is None:
                last_processed = vis.process_frame(frame, args.target_au, vis_mode=args.vis_mode)
            else:
                last_processed["frame_bgr"] = frame

            canvas = compose_view(last_processed, args.target_au, args.vis_mode)

            if writer is None and args.save_path:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))

            if writer is not None:
                writer.write(canvas)

            if args.display:
                cv2.imshow("Adversarial Assistance Visualizer", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    while True:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 in (ord("p"), ord("q")):
                            if k2 == ord("q"):
                                if writer is not None:
                                    writer.release()
                                cv2.destroyAllWindows()
                                return
                            break

            frame_idx += 1
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
