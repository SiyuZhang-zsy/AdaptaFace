from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
import tempfile

# --- 日志系统 ---
logger = logging.getLogger('evaluation_logger')
logger.setLevel(logging.INFO)
logger.propagate = False
file_handler = logging.FileHandler('evaluation_log.txt', encoding='utf-8')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 导入模块 ---
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor
from pgd import PGD

# 是否向前端返回预览图。如果你后面想降延迟，可以改成 False。
ENABLE_PREVIEW = True


def decode_data_url_to_image(image_data_url):
    """把前端传来的 data URL 解码为 OpenCV BGR 图像。"""
    if not image_data_url:
        raise ValueError('缺少 image 数据')

    image_data = re.sub(r'^data:image/.+;base64,', '', image_data_url)
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_np is None:
        raise ValueError('图像解码失败，前端传入的图片数据无效')

    return img_np


def encode_image_to_data_url(img_np, image_format='.jpg'):
    """把 OpenCV BGR 图像重新编码为 data URL，便于前端直接预览。"""
    if not ENABLE_PREVIEW or img_np is None:
        return None

    if not isinstance(img_np, np.ndarray):
        raise TypeError(f'不支持的图像类型: {type(img_np)}')

    success, encoded_img = cv2.imencode(image_format, img_np)
    if not success:
        raise ValueError('图像编码失败，无法生成预览图像')

    base64_bytes = base64.b64encode(encoded_img.tobytes())
    base64_str = base64_bytes.decode('utf-8')
    mime = 'image/jpeg' if image_format.lower() in {'.jpg', '.jpeg'} else 'image/png'
    return f'data:{mime};base64,{base64_str}'


def visualize_perturbation(orig_bgr, adv_bgr):
    """
    将 adv-orig 的扰动可视化到 0-255，便于前端实时查看。
    逻辑与你之前论文图常用的 delta 归一化一致：
    vis = (delta / (2 * max_abs)) + 0.5
    """
    if orig_bgr is None or adv_bgr is None:
        return None

    if orig_bgr.shape != adv_bgr.shape:
        adv_bgr = cv2.resize(adv_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

    delta = adv_bgr.astype(np.float32) - orig_bgr.astype(np.float32)
    max_abs = float(np.max(np.abs(delta)))

    if max_abs < 1e-8:
        vis = np.full_like(orig_bgr, 127, dtype=np.uint8)
    else:
        vis_float = (delta / (2.0 * max_abs)) + 0.5
        vis = np.clip(vis_float * 255.0, 0, 255).astype(np.uint8)

    return vis


class AdversarialAdapter:
    def __init__(self, face_model_path, multitask_model_path, device='cpu'):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.face_detector = FaceDetector(model_path=face_model_path, device=self.device)
        self.multitask_model = MultitaskPredictor(model_path=multitask_model_path, device=self.device)
        self.multitask_model.model.eval()
        self.attack = PGD(self.multitask_model.model, eps=8 / 255, alpha=3 / 255, steps=3)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])
        self.au_list = {"AU1": 0, "AU2": 1, "AU4": 2, "AU6": 3, "AU9": 4, "AU12": 5, "AU25": 6, "AU26": 7}
        self.index_to_name = {v: k for k, v in self.au_list.items()}

    def _save_temp_image_if_needed(self, frame):
        """
        某些 FaceDetector.get_face() 实现内部会调用 cv2.imread，要求输入必须是图片路径。
        如果当前传入的是 numpy 图像，则先临时写盘，再把路径传给 detector。
        """
        if frame is None:
            raise ValueError('输入 frame 为 None')

        if isinstance(frame, (str, os.PathLike)):
            return str(frame), None

        if not isinstance(frame, np.ndarray):
            raise TypeError(f'不支持的 frame 类型: {type(frame)}')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()

        success = cv2.imwrite(temp_path, frame)
        if not success:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise ValueError('无法把输入帧写入临时图片文件')

        return temp_path, temp_path

    def _extract_cropped_face(self, frame):
        temp_path_to_delete = None
        try:
            detector_input, temp_path_to_delete = self._save_temp_image_if_needed(frame)
            face_result = self.face_detector.get_face(detector_input)

            if face_result is None:
                return None

            if isinstance(face_result, (tuple, list)):
                if len(face_result) == 0:
                    return None
                cropped_face = face_result[0]
            else:
                cropped_face = face_result

            return cropped_face
        finally:
            if temp_path_to_delete and os.path.exists(temp_path_to_delete):
                try:
                    os.remove(temp_path_to_delete)
                except OSError as cleanup_error:
                    logger.warning(f'删除临时图片失败: {cleanup_error}')

    def get_au_vector(self, frame):
        try:
            cropped_face = self._extract_cropped_face(frame)
            if cropped_face is None:
                logger.info('提取AU向量失败：未检测到人脸。')
                return None

            _, _, au_output_orig = self.multitask_model.predict(cropped_face)
            return au_output_orig[0].detach().cpu().numpy()
        except Exception as e:
            logger.exception(f'提取AU向量时出错: {e}')
            return None

    def process_frame_for_au_attack(self, frame, attack_enabled: bool, attack_target_au: str, attack_params: dict = None):
        try:
            cropped_face = self._extract_cropped_face(frame)
            if cropped_face is None:
                return {
                    'gesture': 'NoFace',
                    'confidences': {},
                    'baselineConfidences': {},
                    'finalConfidences': {},
                    'faceDetected': False,
                    'attackApplied': False,
                    'croppedFaceImage': None,
                    'adversarialFaceImage': None,
                    'perturbationImage': None,
                }

            _, _, au_output_orig = self.multitask_model.predict(cropped_face)
            baseline_scores = au_output_orig[0]
            baseline_confidences = {self.index_to_name[i]: score.item() for i, score in enumerate(baseline_scores)}
            cropped_face_preview = encode_image_to_data_url(cropped_face)

            if not attack_enabled:
                if attack_target_au is None:
                    recognized_index = torch.argmax(baseline_scores).item()
                    confidence = baseline_scores[recognized_index].item()
                    gesture_name = self.index_to_name.get(recognized_index, 'Unknown') if confidence > 0.7 else 'None'
                elif attack_target_au in self.au_list:
                    target_index = self.au_list[attack_target_au]
                    target_confidence = baseline_scores[target_index].item()
                    gesture_name = attack_target_au if target_confidence > 0.7 else 'None'
                else:
                    gesture_name = 'None'

                return {
                    'gesture': gesture_name,
                    # 保持兼容旧前端：无攻击时这一份就是 baseline
                    'confidences': baseline_confidences,
                    'baselineConfidences': baseline_confidences,
                    'finalConfidences': {},
                    'faceDetected': True,
                    'attackApplied': False,
                    'croppedFaceImage': cropped_face_preview,
                    'adversarialFaceImage': None,
                    'perturbationImage': None,
                }

            if attack_target_au in self.au_list:
                target_index = self.au_list[attack_target_au]
            else:
                target_index = torch.argmax(baseline_scores).item()

            face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
            target_tensor = torch.LongTensor([target_index]).to(self.device)

            if attack_params:
                eps = attack_params.get('eps', 15) / 255.0
                alpha = attack_params.get('alpha', 4) / 255.0
                steps = attack_params.get('steps', 3)
                dynamic_attack = PGD(self.multitask_model.model, eps=eps, alpha=alpha, steps=steps)
                adv_face_tensor = dynamic_attack(face_tensor, target_tensor)
            else:
                adv_face_tensor = self.attack(face_tensor, target_tensor)

            adv_face_numpy = adv_face_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            adv_face_numpy = np.clip(adv_face_numpy * 255.0, 0, 255).astype(np.uint8)
            adv_face_bgr = cv2.cvtColor(adv_face_numpy, cv2.COLOR_RGB2BGR)
            perturbation_vis = visualize_perturbation(cropped_face, adv_face_bgr)

            _, _, au_output_adv = self.multitask_model.predict(adv_face_bgr)
            final_scores = au_output_adv[0]
            final_confidences = {self.index_to_name[i]: score.item() for i, score in enumerate(final_scores)}

            target_au_name = self.index_to_name.get(target_index, 'Unknown')
            score_before = baseline_scores[target_index].item()
            score_after = final_scores[target_index].item()
            logger.info(f'[Adversarial Attack]: Target: {target_au_name} | Score Before: {score_before:.4f} -> Score After: {score_after:.4f}')

            final_recognized_index = torch.argmax(final_scores).item()
            final_confidence = final_scores[final_recognized_index].item()
            final_gesture_name = self.index_to_name.get(final_recognized_index, 'Unknown')
            gesture_name = final_gesture_name if final_confidence > 0.7 else 'None'

            return {
                'gesture': gesture_name,
                # 保持兼容旧前端：有攻击时这一份就是 final
                'confidences': final_confidences,
                'baselineConfidences': baseline_confidences,
                'finalConfidences': final_confidences,
                'faceDetected': True,
                'attackApplied': True,
                'croppedFaceImage': cropped_face_preview,
                'adversarialFaceImage': encode_image_to_data_url(adv_face_bgr),
                'perturbationImage': encode_image_to_data_url(perturbation_vis, image_format='.png') if perturbation_vis is not None else None,
            }
        except Exception as e:
            logger.exception(f'在 process_frame_for_au_attack 中发生错误: {e}')
            return {
                'gesture': 'Error',
                'confidences': {},
                'baselineConfidences': {},
                'finalConfidences': {},
                'faceDetected': False,
                'attackApplied': False,
                'croppedFaceImage': None,
                'adversarialFaceImage': None,
                'perturbationImage': None,
            }


AU_INFO = {
    'AU1': {'name': '眉头上扬'},
    'AU2': {'name': '眉头外扬'},
    'AU4': {'name': '皱眉'},
    'AU6': {'name': '脸颊提肌'},
    'AU9': {'name': '皱鼻'},
    'AU12': {'name': '微笑'},
    'AU25': {'name': '嘴唇张开'},
    'AU26': {'name': '下巴张开'}
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using device: {device}')

adapter = AdversarialAdapter(
    face_model_path='./weights/Alignment_RetinaFace.pth',
    multitask_model_path='./weights/MTL_backbone.pth',
    device=device
)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
gesture_templates = {}
AU_VECTOR_DISTANCE_THRESHOLD = 0.5
session_results = []


@app.route('/')
def index_page():
    global session_results
    if session_results:
        logger.info('=' * 50)
        logger.info('一次完整的评估已结束，结果汇总如下:')
        for i, result in enumerate(session_results):
            log_message = (
                f"  步骤 {i + 1}: \n"
                f"    - 任务: {result.get('taskName', 'N/A')}\n"
                f"    - 模式: {result.get('mode', 'N/A')}\n"
                f"    - 动作名称: {result.get('gestureName', 'N/A')}\n"
                f"    - 条件: {result.get('condition', 'N/A')}\n"
                f"    - 客观性能数据: {result.get('results', {})}"
            )
            logger.info(log_message)
        logger.info('=' * 50)
        session_results = []
    return render_template('index.html')


@app.route('/settings')
def settings_page():
    return render_template('settings.html')


@app.route('/gallery')
def gallery_page():
    return render_template('gallery.html')


@app.route('/piano')
def piano_page():
    return render_template('piano.html')


@app.route('/zoom')
def zoom_page():
    return render_template('zoom.html')


@app.route('/process_frame', methods=['POST'])
def process_frame_api():
    try:
        data = request.json or {}
        attack_enabled = data.get('attackEnabled', False)
        mode = data.get('mode', 'au')
        attack_params = data.get('attackParams', None)
        fingerprint_match_occurred = False

        img_np = decode_data_url_to_image(data.get('image'))
        attack_target = data.get('au') or data.get('target_au')

        if attack_enabled and mode == 'au_vector' and 'default_user' in gesture_templates:
            template = gesture_templates['default_user']
            current_au_vector = adapter.get_au_vector(img_np)

            if current_au_vector is not None:
                distance = np.linalg.norm(template['au_vector'] - current_au_vector)
                if distance < AU_VECTOR_DISTANCE_THRESHOLD:
                    attack_target = template['target_au']
                    fingerprint_match_occurred = True
                    logger.info(f'[AdaptaFace Trigger]: 指纹匹配成功, 攻击目标为 {attack_target} (Distance: {distance:.4f})')
                else:
                    attack_target = None
                    logger.info(f'[AdaptaFace Trigger]: 指纹未匹配，Distance: {distance:.4f}')
            else:
                attack_target = None

        result = adapter.process_frame_for_au_attack(
            img_np,
            attack_enabled=(attack_enabled and attack_target is not None),
            attack_target_au=attack_target,
            attack_params=attack_params
        )

        preview_image = encode_image_to_data_url(img_np)

        log_message = f"[实时识别]: 模式[{'AdaptaFace' if attack_enabled else '基线'}], 目标[{data.get('au') or data.get('target_au')}], 结果[{result['gesture']}]"
        logger.info(log_message)

        return jsonify({
            'status': 'success',
            'gesture': result['gesture'],
            'confidences': result['confidences'],
            'baselineConfidences': result.get('baselineConfidences', {}),
            'finalConfidences': result.get('finalConfidences', {}),
            'fingerprintMatch': fingerprint_match_occurred,
            'previewImage': preview_image,
            'faceDetected': result['faceDetected'],
            'attackApplied': result['attackApplied'],
            'croppedFaceImage': result['croppedFaceImage'],
            'adversarialFaceImage': result['adversarialFaceImage'],
            'perturbationImage': result['perturbationImage'],
        })
    except Exception as e:
        logger.exception(f'在 /process_frame 中发生错误: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/create_au_vector_template', methods=['POST'])
def create_au_vector_template_api():
    try:
        data = request.json or {}
        target_au = data.get('target_au')
        gesture_frames = data.get('gesture_frames', [])
        logger.info(f'[模板]: 收到 {len(gesture_frames)} 帧用于创建模板。')

        if not gesture_frames:
            return jsonify({'status': 'error', 'message': '没有收到录制帧，请检查前端传参 gesture_frames。'}), 400

        au_vector_fingerprint = None

        for i, frame_url in enumerate(gesture_frames):
            try:
                img = decode_data_url_to_image(frame_url)
                au_vector = adapter.get_au_vector(img)
                if au_vector is not None:
                    au_vector_fingerprint = au_vector
                    logger.info(f'[模板]: 第 {i + 1} 帧成功提取 AU 向量指纹。')
                    break
                logger.warning(f'[模板]: 第 {i + 1} 帧未能提取 AU 向量。')
            except Exception as frame_error:
                logger.exception(f'[模板]: 第 {i + 1} 帧处理失败: {frame_error}')

        if au_vector_fingerprint is None:
            logger.error('[模板]: 连拍组识别全部失败。')
            return jsonify({'status': 'error', 'message': '模板创建失败：未能从任何录制帧中提取到有效人脸/AU向量。'}), 400

        gesture_templates['default_user'] = {
            'au_vector': au_vector_fingerprint,
            'target_au': target_au
        }
        formatted_vector = [f'{val:.3f}' for val in au_vector_fingerprint]
        logger.info(f'[模板]: 成功创建AU向量模板, 目标AU为 {target_au}, 指纹向量: {formatted_vector}')

        return jsonify({'status': 'success', 'message': '模板创建成功!'})
    except Exception as e:
        logger.exception(f'在 /create_au_vector_template 中发生错误: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/analyze_and_create_template', methods=['POST'])
def analyze_and_create_template_api():
    try:
        data = request.json or {}
        gesture_frames = data.get('gesture_frames', [])
        logger.info(f'[智能分析]: 收到 {len(gesture_frames)} 帧用于自动分析模板。')

        if not gesture_frames:
            return jsonify({'status': 'error', 'message': '没有收到录制帧，请检查前端传参 gesture_frames。'}), 400

        au_vector_fingerprint = None

        for i, frame_url in enumerate(gesture_frames):
            try:
                img = decode_data_url_to_image(frame_url)
                au_vector = adapter.get_au_vector(img)
                if au_vector is not None:
                    au_vector_fingerprint = au_vector
                    logger.info(f'[智能分析]: 第 {i + 1} 帧成功提取 AU 向量。')
                    break
                logger.warning(f'[智能分析]: 第 {i + 1} 帧未能提取 AU 向量。')
            except Exception as frame_error:
                logger.exception(f'[智能分析]: 第 {i + 1} 帧处理失败: {frame_error}')

        if au_vector_fingerprint is None:
            logger.error('[智能分析]: 连拍组识别全部失败。')
            return jsonify({'status': 'error', 'message': '动作录制失败：未能从任何录制帧中提取到有效人脸/AU向量。'}), 400

        dominant_au_index = int(np.argmax(au_vector_fingerprint))
        determined_target_au = adapter.index_to_name.get(dominant_au_index, 'Unknown')

        formatted_vector = [f'{val:.3f}' for val in au_vector_fingerprint]
        logger.info(f'[智能分析]: 用户动作中最显著的特征是 {determined_target_au} (索引: {dominant_au_index}), 向量值: {formatted_vector}')

        gesture_templates['default_user'] = {
            'au_vector': au_vector_fingerprint,
            'target_au': determined_target_au
        }

        return jsonify({
            'status': 'success',
            'message': '模板创建成功!',
            'determined_au': determined_target_au,
            'determined_au_name': AU_INFO.get(determined_target_au, {}).get('name', '未知动作')
        })
    except Exception as e:
        logger.exception(f'在 /analyze_and_create_template 中发生错误: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/log_results', methods=['POST'])
def log_results_api():
    global session_results
    try:
        data = request.json or {}
        session_results.append(data)
        log_message = (
            f"[客观数据记录]: \n"
            f"    - 任务: {data.get('taskName', 'N/A')}\n"
            f"    - 模式: {data.get('mode', 'N/A')}\n"
            f"    - 动作名称: {data.get('gestureName', 'N/A')}\n"
            f"    - 条件: {data.get('condition', 'N/A')}\n"
            f"    - 结果: {data.get('results', {})}"
        )
        logger.info(log_message)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception(f'在 /log_results 中发生错误: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/log_feedback', methods=['POST'])
def log_feedback_api():
    try:
        data = request.json or {}
        log_message = (
            f"[主观反馈记录]: \n"
            f"    - 任务: {data.get('taskName', 'N/A')}\n"
            f"    - 模式: {data.get('mode', 'N/A')}\n"
            f"    - 动作名称: {data.get('gestureName', 'N/A')}\n"
            f"    - 条件: {data.get('condition', 'N/A')}\n"
            f"    - 评分: {data.get('feedback', {})}"
        )
        logger.info(log_message)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception(f'在 /log_feedback 中发生错误: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)
