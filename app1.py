from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
from collections import Counter
import logging
from datetime import datetime
import os

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

# --- AdversarialAdapter 类 ---
class AdversarialAdapter:
    def __init__(self, face_model_path, multitask_model_path, device='cpu'):
        if device == 'cuda' and torch.cuda.is_available(): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')
        self.face_detector = FaceDetector(model_path=face_model_path, device=self.device)
        self.multitask_model = MultitaskPredictor(model_path=multitask_model_path, device=self.device)
        self.multitask_model.model.eval()
        self.attack = PGD(self.multitask_model.model, eps=8/255, alpha=3/255, steps=3)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True)])
        self.au_list = { "AU1":0, "AU2":1, "AU4":2, "AU6":3, "AU9":4, "AU12":5, "AU25":6, "AU26":7 }
        self.index_to_name = {v: k for k, v in self.au_list.items()}

    def get_au_vector(self, frame):
        try:
            cropped_face, _, _ = self.face_detector.get_face(frame)
            if cropped_face is None: return None
            _, _, au_output_orig = self.multitask_model.predict(cropped_face)
            return au_output_orig[0].cpu().numpy()
        except Exception as e:
            logger.error(f"提取AU向量时出错: {e}")
            return None

    def process_frame_for_au_attack(self, frame, attack_enabled: bool, attack_target_au: str, attack_params: dict = None):
        cropped_face, _, _ = self.face_detector.get_face(frame)
        if cropped_face is None: return ("NoFace", {})
        
        _, _, au_output_orig = self.multitask_model.predict(cropped_face)
        baseline_scores = au_output_orig[0]
        confidences_dict = {self.index_to_name[i]: score.item() for i, score in enumerate(baseline_scores)}

        if not attack_enabled:
            if attack_target_au is None:
                 recognized_index = torch.argmax(baseline_scores).item()
                 confidence = baseline_scores[recognized_index].item()
                 return (self.index_to_name.get(recognized_index, "Unknown") if confidence > 0.7 else "None", confidences_dict)
            if attack_target_au in self.au_list:
                target_index = self.au_list[attack_target_au]
                target_confidence = baseline_scores[target_index].item()
                if target_confidence > 0.7: return (attack_target_au, confidences_dict)
                else: return ("None", confidences_dict)
            return ("None", confidences_dict)

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
        adv_face_bgr = cv2.cvtColor(adv_face_numpy, cv2.COLOR_RGB2BGR)
        _, _, au_output_adv = self.multitask_model.predict(adv_face_bgr)
        final_scores = au_output_adv[0]

        # ▼▼▼ 新添加的日志记录代码 ▼▼▼
        target_au_name = self.index_to_name.get(target_index, "Unknown")
        score_before = baseline_scores[target_index].item()
        score_after = final_scores[target_index].item()
        logger.info(f"[Adversarial Attack]: Target: {target_au_name} | Score Before: {score_before:.4f} -> Score After: {score_after:.4f}")
        # ▲▲▲ 新添加的日志记录代码 ▲▲▲

        final_recognized_index = torch.argmax(final_scores).item()
        final_confidence = final_scores[final_recognized_index].item()
        final_gesture_name = self.index_to_name.get(final_recognized_index, "Unknown")
        adv_confidences_dict = {self.index_to_name[i]: score.item() for i, score in enumerate(final_scores)}
        
        if final_confidence > 0.7: return (final_gesture_name, adv_confidences_dict)
        else: return ("None", adv_confidences_dict)

AU_INFO = { "AU1":{"name":"眉头上扬"},"AU2":{"name":"眉头外扬"},"AU4":{"name":"皱眉"},"AU6":{"name":"脸颊提肌"},"AU9":{"name":"皱鼻"},"AU12":{"name":"微笑"},"AU25":{"name":"嘴唇张开"},"AU26":{"name":"下巴张开"} }

# --- 初始化 ---
adapter = AdversarialAdapter(
    face_model_path='./weights/Alignment_RetinaFace.pth',
    multitask_model_path='./weights/MTL_backbone.pth',
    device='cpu'
)
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
gesture_templates = {} 
AU_VECTOR_DISTANCE_THRESHOLD = 0.5 
session_results = []

# --- 网页路由 ---
@app.route('/')
def index_page():
    global session_results
    if session_results:
        logger.info("="*50)
        logger.info("一次完整的评估已结束，结果汇总如下:")
        for i, result in enumerate(session_results):
            log_message = (
                f"  步骤 {i+1}: \n"
                f"    - 任务: {result.get('taskName', 'N/A')}\n"
                f"    - 模式: {result.get('mode', 'N/A')}\n"
                f"    - 动作名称: {result.get('gestureName', 'N/A')}\n"
                f"    - 条件: {result.get('condition', 'N/A')}\n"
                f"    - 客观性能数据: {result.get('results', {})}"
            )
            logger.info(log_message)
        logger.info("="*50)
        session_results = []
    return render_template('index.html')

@app.route('/settings')
def settings_page(): return render_template('settings.html')
@app.route('/gallery')
def gallery_page(): return render_template('gallery.html')
@app.route('/piano')
def piano_page(): return render_template('piano.html')
@app.route('/zoom')
def zoom_page(): return render_template('zoom.html')

# --- API 接口 ---
@app.route('/process_frame', methods=['POST'])
def process_frame_api():
    try:
        data = request.json
        attack_enabled = data.get('attackEnabled', False)
        mode = data.get('mode', 'au')
        attack_params = data.get('attackParams', None)
        fingerprint_match_occurred = False

        image_data_url = data.get('image')
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        attack_target = data.get('au') or data.get('target_au')

        if attack_enabled and mode == 'au_vector' and 'default_user' in gesture_templates:
            template = gesture_templates['default_user']
            current_au_vector = adapter.get_au_vector(img_np)
            
            if current_au_vector is not None:
                distance = np.linalg.norm(template['au_vector'] - current_au_vector)
                if distance < AU_VECTOR_DISTANCE_THRESHOLD:
                    attack_target = template['target_au']
                    fingerprint_match_occurred = True
                    logger.info(f"[AdaptaFace Trigger]: 指纹匹配成功, 攻击目标为 {attack_target} (Distance: {distance:.4f})")
                else: # 如果指纹不匹配，则本次不进行攻击
                    attack_target = None
            else: # 如果没检测到脸，也不攻击
                attack_target = None
        
        result_tuple = adapter.process_frame_for_au_attack(img_np, attack_enabled=(attack_enabled and attack_target is not None), attack_target_au=attack_target, attack_params=attack_params)
        
        log_message = f"[实时识别]: 模式[{'AdaptaFace' if attack_enabled else '基线'}], 目标[{data.get('au') or data.get('target_au')}], 结果[{result_tuple[0]}]"
        logger.info(log_message)
        return jsonify({'status': 'success', 'gesture': result_tuple[0], 'confidences': result_tuple[1], 'fingerprintMatch': fingerprint_match_occurred})
    except Exception as e:
        logger.error(f"在 /process_frame 中发生错误: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/create_au_vector_template', methods=['POST'])
def create_au_vector_template_api():
    try:
        data = request.json
        target_au = data.get('target_au')
        gesture_frames = data.get('gesture_frames', [])
        au_vector_fingerprint = None
        for frame_url in gesture_frames:
            img_data = re.sub('^data:image/.+;base64,', '', frame_url)
            image_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            au_vector = adapter.get_au_vector(img)
            if au_vector is not None:
                au_vector_fingerprint = au_vector
                logger.info("[模板]: 在连拍照片中成功提取AU向量指纹。")
                break 
        if au_vector_fingerprint is None:
            logger.error("[模板]: 连拍组识别全部失败。")
            return jsonify({'status': 'error', 'message': '动作识别失败，请确保动作清晰、正对摄像头。'}), 400
        gesture_templates['default_user'] = {
            'au_vector': au_vector_fingerprint,
            'target_au': target_au
        }
        formatted_vector = [f"{val:.3f}" for val in au_vector_fingerprint]
        logger.info(f"[模板]: 成功创建AU向量模板, 目标AU为 {target_au}, 指纹向量: {formatted_vector}")
        
        return jsonify({ 'status': 'success', 'message': '模板创建成功!' })
    except Exception as e:
        logger.error(f"在 /create_au_vector_template 中发生错误: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analyze_and_create_template', methods=['POST'])
def analyze_and_create_template_api():
    try:
        data = request.json
        gesture_frames = data.get('gesture_frames', [])
        au_vector_fingerprint = None
        for frame_url in gesture_frames:
            img_data = re.sub('^data:image/.+;base64,', '', frame_url)
            image_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            au_vector = adapter.get_au_vector(img)
            if au_vector is not None:
                au_vector_fingerprint = au_vector
                break 
        
        if au_vector_fingerprint is None:
            logger.error("[智能分析]: 连拍组识别全部失败。")
            return jsonify({'status': 'error', 'message': '动作录制失败，请确保动作清晰、正对摄像头。'}), 400

        dominant_au_index = np.argmax(au_vector_fingerprint)
        determined_target_au = adapter.index_to_name.get(dominant_au_index, "Unknown")
        
        formatted_vector = [f"{val:.3f}" for val in au_vector_fingerprint]
        logger.info(f"[智能分析]: 用户动作中最显著的特征是 {determined_target_au} (索引: {dominant_au_index}), 向量值: {formatted_vector}")

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
        logger.error(f"在 /analyze_and_create_template 中发生错误: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/log_results', methods=['POST'])
def log_results_api():
    global session_results
    try:
        data = request.json
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
        logger.error(f"在 /log_results 中发生错误: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/log_feedback', methods=['POST'])
def log_feedback_api():
    try:
        data = request.json
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
        logger.error(f"在 /log_feedback 中发生错误: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- 启动服务器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)