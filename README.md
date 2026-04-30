# AdaptaFace
AdaptaFace is a research prototype for input-level adversarial assistance in facial-gesture interaction. It inserts a compatibility layer between a user's facial input and an existing facial-action recognizer, allowing user-compatible gestures to be translated into recognizer-compatible inputs without retraining the downstream model.

The current release support using OpenFace 3.0 and OpenGraphAU-compatible models as external facial-analysis backends. These models are used to generate and evaluate input-level perturbations. AdaptaFace will support more models in future.

---
## Intro

[AdaptaFace.pdf](https://github.com/user-attachments/files/27253581/fig11.1.pdf)

---
## Video —— Intro

https://github.com/user-attachments/assets/5b073703-70c3-40b1-89f7-6d75424018b9

---
## Function 1 —— Enhances Gesture Recognition

https://github.com/user-attachments/assets/db47919f-44e3-4de4-9209-ae8394a55971

---
## Function 2 —— Vocabulary Extension

https://github.com/user-attachments/assets/a4e1be16-fb3a-495e-a7e0-05e60f0e8a06

---

## Overview

AdaptaFace explores a third personalization logic for facial-gesture interaction:

1. Users do not need to fully adapt their movements to recognizer-defined canonical gestures.
2. The downstream recognizer does not need to be retrained for every user.
3. Instead, an input-level compatibility layer applies bounded adversarial assistance to make user-compatible facial input more legible to an existing recognizer.

This repository includes:

- the local Flask backend for the interactive prototype;
- browser-based study/demo interfaces;
- PGD-based adversarial assistance code;
- a real-time demo visualizer;
- offline validation scripts for cross-dataset, cross-model, and model-transfer analyses;
- selected-case export utilities for figure-style examples.

---

## Repository structure

The current repository is expected to be organized as follows:


```text
AdaptaFace/
  README.md
  .gitignore

  app.py
  pgd.py
  realtime_attack_visualizer.py

  templates/
    index.html
    settings.html
    gallery.html
    piano.html
    zoom.html

  weights/
    README.md

  eval_affwild2_demo_state_split_v2.py
  eval_opengraphau_self_attack_fixed.py
  eval_cross_model_opengraphau_to_openface.py
  export_selected_cases_epsnorm.py
```

Suggested local-only folders:

```text
data/          # External datasets, not committed
outputs/       # Generated outputs, not committed
weights/       # Local external model weights, not committed except README.md
```

---

## External model backends

The current release uses two external facial-analysis backends.

### OpenFace 3.0

OpenFace 3.0 is used as the OpenFace-compatible face detection and AU-recognition backend.

Upstream repository:

```text
https://github.com/CMU-MultiComp-Lab/OpenFace-3.0
```

AdaptaFace expects the following Python imports to work:

```bash
python -c "from openface.face_detection import FaceDetector; from openface.multitask_model import MultitaskPredictor; print('OpenFace backend ready')"
```

The local prototype and demo visualizer expect OpenFace-compatible weights under:

```text
weights/
  Alignment_RetinaFace.pth
  MTL_backbone.pth
```

These files are not redistributed in this repository. Please install OpenFace 3.0 and download its weights following the upstream instructions. After downloading, copy or symlink the required files into `./weights/`.

### OpenGraphAU

OpenGraphAU is used as an additional AU-recognition backend for cross-model and model-transfer offline validation.

Upstream repository used for OpenGraphAU-compatible experiments:

```text
https://github.com/lingjivoo/OpenGraphAU
```

If your checkpoint comes from a different OpenGraphAU or ME-GraphAU-compatible fork, please use the corresponding upstream repository instead.

The offline validation scripts do not assume a fixed OpenGraphAU installation path. Instead, pass the local paths using:

```bash
--og_repo /path/to/OpenGraphAU
--og_checkpoint /path/to/opengraphau_checkpoint.pth
```

---

## Supported model backends

The current release provides reference implementations based on OpenFace 3.0 and OpenGraphAU-compatible models. AdaptaFace itself is designed as a model-agnostic compatibility layer: users may add support for other differentiable facial-analysis models by implementing a backend wrapper that exposes prediction outputs and gradients.

OpenFace 3.0 and OpenGraphAU are used as external recognizer/proxy backends. They are not redistributed as part of this repository, and their source code and model assets are subject to their own licenses.

---

## Environment setup

A typical Python environment is:

```bash
conda create -n adaptaface python=3.11
conda activate adaptaface
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision
```

Then install the main Python dependencies:

```bash
pip install flask flask-cors opencv-python numpy pan
das pillow scikit-learn tqdm
```

For the real-time screen-capture demo, also install:

```bash
pip install mss
```

Install OpenFace 3.0 following the upstream instructions:

```bash
pip install openface-test
openface download
```

Then verify that the OpenFace backend is importable:

```bash
python -c "from openface.face_detection import FaceDetector; from openface.multitask_model import MultitaskPredictor; print('OpenFace backend ready')"
```

---

## Model weights

This repository does not redistribute third-party model weights.

Expected local structure:

```text
weights/
  Alignment_RetinaFace.pth
  MTL_backbone.pth
```

These files are required by the local prototype and real-time visualizer.

You may either copy the weights:

```bash
mkdir -p weights
cp /path/to/openface_weights/Alignment_RetinaFace.pth weights/
cp /path/to/openface_weights/MTL_backbone.pth weights/
```

or create symbolic links:

```bash
mkdir -p weights
ln -s /path/to/openface_weights/Alignment_RetinaFace.pth weights/Alignment_RetinaFace.pth
ln -s /path/to/openface_weights/MTL_backbone.pth weights/MTL_backbone.pth
```

On Windows PowerShell, symbolic links can be created with:

```powershell
New-Item -ItemType SymbolicLink -Path .\weights\Alignment_RetinaFace.pth -Target "C:\path\to\Alignment_RetinaFace.pth"
New-Item -ItemType SymbolicLink -Path .\weights\MTL_backbone.pth -Target "C:\path\to\MTL_backbone.pth"
```

Recommended `weights/README.md`:

```markdown
# OpenFace 3.0 Weights

This folder is reserved for local OpenFace-3.0-compatible model weights required by the AdaptaFace prototype.

Expected local structure:

```text
weights/
  Alignment_RetinaFace.pth
  MTL_backbone.pth
```

These files are not included in this repository. AdaptaFace uses OpenFace 3.0 as an external AU-recognition backend, and OpenFace 3.0 model assets are subject to their own license terms.

Please install OpenFace 3.0 and download its weights following the upstream OpenFace 3.0 instructions. After downloading, copy or symlink the required files into this folder.

This repository provides the AdaptaFace compatibility-layer code and evaluation scripts, but does not redistribute OpenFace 3.0 model weights.
```

---

## Running the local prototype

Start the local Flask backend:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Available pages include:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/settings
http://127.0.0.1:5000/gallery
http://127.0.0.1:5000/piano
http://127.0.0.1:5000/zoom
```

The browser will request camera access. For local testing, use `127.0.0.1` or `localhost`, because browsers usually allow webcam access in local secure contexts.

The backend writes local logs to:

```text
evaluation_log.txt
```

Do not commit this file, especially if it contains user-study or participant-related records.

---

## Optional remote access with ngrok

The original remote user-study deployment used ngrok so that participants could access the browser interface remotely while computation remained on the experimenter's local machine.

For artifact review and local testing, ngrok is **not required**.

If remote testing is needed, first start the Flask backend:

```bash
python app.py
```

Then expose port 5000:

```bash
ngrok http 5000
```

Open the generated HTTPS URL on the remote device.

Do not commit ngrok tokens, private URLs, participant links, or `.env` files to the repository.

---

## Real-time demo visualizer

The standalone visualizer processes either a video file or a screen capture. It displays the original input, face crop, adversarially assisted face, perturbation visualization, and AU confidence changes.

### Video input

```bash
python realtime_attack_visualizer.py \
  --source video \
  --video_path ./demo/input.mp4 \
  --target_au AU12 \
  --device cuda \
  --display \
  --save_path ./outputs/demo_AU12.mp4
```

### Screen input

```bash
python realtime_attack_visualizer.py \
  --source screen \
  --target_au AU12 \
  --monitor 1 \
  --device cuda \
  --display
```

Supported OpenFace AU targets:

```text
AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26
```

Default perturbation parameters:

```text
epsilon = 8/255
alpha   = 3/255
steps   = 3
```

---

## Offline validation

The offline scripts evaluate the adversarial-assistance mechanism on Aff-Wild2-style AU annotations and aligned face images.

Expected dataset layout:

```text
data/AffWild2/
  Validation_Set/
    *.txt
  cropped_aligned_images/
    video_id/
      00001.jpg
      00002.jpg
      ...
```

The scripts parse AU annotation files, build frame-level indices, filter AU source/target states, apply targeted perturbations, and write summary outputs.

Main AU state conditions:

```text
10 = source1_target0  # substitution setting
01 = source0_target1  # enhancement setting
```

Other supported states:

```text
11 = source1_target1
00 = source0_target0
all
```

The main shared AU set is:

```text
AU1, AU2, AU4, AU6, AU12, AU25, AU26
```

---

## Cross-dataset validation: OpenFace -> OpenFace

This evaluates OpenFace-generated perturbations on Aff-Wild2 using the OpenFace evaluator.

```bash
python eval_affwild2_demo_state_split_v2.py \
  --data_root ./data/AffWild2 \
  --model_path ./weights/MTL_backbone.pth \
  --out_dir ./outputs/cross_dataset_openface \
  --source_au all \
  --target_au all \
  --state all \
  --batch_size 8 \
  --num_workers 4 \
  --image_size 224 \
  --sample_every 1 \
  --eps 0.031372549 \
  --alpha 0.011764706 \
  --steps 3 \
  --confidence_threshold 0.7 \
  --min_eligible_frames 20 \
  --device cuda
```

Expected outputs:

```text
outputs/cross_dataset_openface/
  summary.json
  pair_state_summary.csv
  frame_level_outputs.csv
  skipped_small_pair_states.csv
```

---

## Cross-model validation: OpenGraphAU -> OpenGraphAU

This evaluates the same assistance logic using OpenGraphAU as both the perturbation generator and evaluator.

```bash
python eval_opengraphau_self_attack_fixed.py \
  --data_root ./data/AffWild2 \
  --out_dir ./outputs/cross_model_opengraphau \
  --source_au all \
  --target_au all \
  --state all \
  --batch_size 8 \
  --num_workers 4 \
  --image_size 224 \
  --sample_every 1 \
  --eps 0.031372549 \
  --alpha 0.011764706 \
  --steps 3 \
  --confidence_threshold 0.7 \
  --pass_threshold 0.5 \
  --min_eligible_frames 20 \
  --device cuda \
  --og_repo /path/to/OpenGraphAU \
  --og_checkpoint /path/to/opengraphau_checkpoint.pth \
  --og_device cuda \
  --og_arc resnet50 \
  --og_stage 2 \
  --og_neighbor_num 4 \
  --og_metric dots \
  --og_apply_official_resize
```

Expected outputs:

```text
outputs/cross_model_opengraphau/
  summary.json
  pair_state_summary.csv
  frame_level_outputs.csv
  skipped_small_pair_states.csv
```

---

## Model-transfer validation: OpenGraphAU -> OpenFace

This setting generates perturbations with OpenGraphAU and evaluates the assisted images using OpenFace.

```bash
python eval_cross_model_opengraphau_to_openface.py \
  --data_root ./data/AffWild2 \
  --model_path ./weights/MTL_backbone.pth \
  --out_dir ./outputs/model_transfer_opengraphau_to_openface \
  --source_au all \
  --target_au all \
  --state all \
  --batch_size 8 \
  --num_workers 4 \
  --image_size 224 \
  --sample_every 1 \
  --eps 0.031372549 \
  --alpha 0.011764706 \
  --steps 3 \
  --confidence_threshold 0.7 \
  --pass_threshold 0.5 \
  --min_eligible_frames 20 \
  --device cuda \
  --og_repo /path/to/OpenGraphAU \
  --og_checkpoint /path/to/opengraphau_checkpoint.pth \
  --og_device cuda \
  --og_arc resnet50 \
  --og_stage 2 \
  --og_neighbor_num 4 \
  --og_metric dots \
  --og_apply_official_resize
```

Expected outputs:

```text
outputs/model_transfer_opengraphau_to_openface/
  summary.json
  pair_state_summary.csv
  frame_level_outputs.csv
  skipped_small_pair_states.csv
```

---

## Exporting selected figure examples

`export_selected_cases_epsnorm.py` exports figure-ready examples, including:

```text
original image
assisted image
epsilon-normalized perturbation visualization
metadata.json
export_manifest.csv
export_manifest.json
```

Example command:

```bash
python export_selected_cases_epsnorm.py \
  --case_list ./selected_cases.csv \
  --export_root ./outputs/selected_examples \
  --data_root ./data/AffWild2 \
  --crossdataset_root ./outputs/cross_dataset_openface \
  --crossmodel_root ./outputs/cross_model_opengraphau \
  --transfer_root ./outputs/model_transfer_opengraphau_to_openface \
  --openface_model ./weights/MTL_backbone.pth \
  --og_repo /path/to/OpenGraphAU \
  --og_checkpoint /path/to/opengraphau_checkpoint.pth \
  --image_size 224 \
  --eps 0.031372549 \
  --alpha 0.011764706 \
  --steps 3 \
  --conf_threshold 0.7 \
  --device cuda \
  --attack_device cuda \
  --eval_device cuda \
  --og_stage 2 \
  --og_arc resnet50 \
  --og_neighbor_num 4 \
  --og_metric dots \
  --og_apply_official_resize
```

The case list should be a CSV or XLSX file containing at least:

```text
case_id
setting
target
video_id
frame_id
image_path
```

Optional columns:

```text
source
state
baseline_conf
final_conf
gain
```

Supported settings:

```text
cross-dataset
cross-model
transfer-dataset
```

---

## Relation to reported offline results

The aggregate offline validation results reported in the paper were computed from the generated `summary.json` and `pair_state_summary.csv` files for each validation setting:

```text
outputs/cross_dataset_openface/
outputs/cross_model_opengraphau/
outputs/model_transfer_opengraphau_to_openface/
```

No separate table-formatting script is required. The scripts above produce the summary files from which aggregate offline validation results can be checked.

---

## Data and privacy

The user study involved participants with motor impairments and webcam-based facial interaction. For privacy and ethics reasons, this repository does not release:

- raw participant videos;
- raw webcam frames;
- participant-identifiable logs;
- full interview transcripts;
- consent forms;
- recruitment messages;
- private deployment URLs.

Generated logs such as `evaluation_log.txt` should be treated as local experiment records and should not be committed.

---

## Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Local logs
*.log
evaluation_log.txt

# Local outputs
outputs/
results/

# External datasets
data/
datasets/

# External OpenFace/OpenGraphAU source or local experiments
openface/
openface_backup/
OpenGraphAU/

# External model weights
weights/*.pth
weights/*.pt
weights/*.pkl
weights/*.tar
weights/*.ckpt
weights/._*
weights/.DS_Store
!weights/README.md

# OS/editor files
.DS_Store
._*
.vscode/
.idea/

# Local environment or private deployment files
.env
.ngrok*
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'openface'`

Install OpenFace 3.0 following the upstream instructions, then verify:

```bash
python -c "from openface.face_detection import FaceDetector; from openface.multitask_model import MultitaskPredictor; print('OpenFace backend ready')"
```

### Missing OpenFace weights

Make sure the following files exist:

```text
weights/Alignment_RetinaFace.pth
weights/MTL_backbone.pth
```

### Webcam does not open

Use:

```text
http://127.0.0.1:5000
```

instead of a non-local HTTP address. Browsers may block webcam access on non-HTTPS remote addresses.

### CUDA is not available

Use CPU mode where supported:

```bash
--device cpu
```

or let the code fall back to CPU automatically.

### OpenGraphAU import error

Make sure `--og_repo` points to the root of the OpenGraphAU-compatible repository and that `--og_checkpoint` points to a valid checkpoint:

```bash
--og_repo /path/to/OpenGraphAU
--og_checkpoint /path/to/opengraphau_checkpoint.pth
```

---

## License

This repository contains the AdaptaFace research prototype code.

OpenFace 3.0, OpenGraphAU, their source code, and their model weights are external dependencies and are subject to their own licenses. This repository does not redistribute those external model assets.

Please check the upstream licenses before using or redistributing any third-party components.

