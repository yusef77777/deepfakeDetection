# Deepfake Detection App — XceptionNet Video Forensics

[![Releases](https://img.shields.io/badge/Releases-v1.0-orange)](https://github.com/yusef77777/deepfakeDetection/releases) [![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9C%93-EE4C2C)](https://pytorch.org/) [![Django](https://img.shields.io/badge/Django-%E2%9C%93-092E20)](https://www.djangoproject.com/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

Detect deepfake videos with a Django web interface and a fine-tuned XceptionNet (Extreme Inception) model. Upload video files, run inference on sampled frames, and get clear per-video and per-frame scores. Visit the releases page to download the packaged model and app: https://github.com/yusef77777/deepfakeDetection/releases

Preview
![Deepfake demo image](https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=6b482b22dd3e2f9d3f4c8b3f17d0d5a8)

Table of contents
- Features
- Release / Download
- Quick start
- Run locally (development)
- Run with Docker
- Model and architecture
- Inference pipeline
- Web UI and API
- Training and dataset
- Evaluation and metrics
- File structure
- Contributing
- License
- Acknowledgements

Features
- Web upload and batch upload support via a simple Django app. 🎥
- Fine-tuned XceptionNet model (PyTorch) for face-level deepfake detection. 🤖
- Frame extraction and face cropping pipeline with confident predictions per-frame and aggregated per-video scores. 🧠
- REST API endpoints for programmatic use. 🔌
- Dockerfile and docker-compose setup for production deploys. 📦
- Lightweight frontend with status updates and result export (CSV / JSON). 📄

Release / Download
- Visit the Releases page and download the release package. The package contains the trained model, a packaged app, and an installer script.
- The release link is: https://github.com/yusef77777/deepfakeDetection/releases
- Example release asset name (packaged in releases): deepfakeDetection-v1.0.0-linux.tar.gz
- After downloading the release asset, extract it and run the included installer:
  - tar -xzf deepfakeDetection-v1.0.0-linux.tar.gz
  - cd deepfakeDetection-v1.0.0
  - ./install.sh
- The installer places model weights in models/xception_finetuned.pt and config files in conf/.

Quick start (local, minimal)
Prerequisites
- Python 3.9+ (3.10 recommended)
- PyTorch 1.10+ with CUDA if you have a GPU
- FFmpeg (for video frame extraction)
- Git

Clone and install
- git clone https://github.com/yusef77777/deepfakeDetection.git
- cd deepfakeDetection
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt

Model and release
- Download the release asset from the Releases page: https://github.com/yusef77777/deepfakeDetection/releases
- Extract and copy the model to the models/ folder:
  - cp /path/to/deepfakeDetection-v1.0.0/models/xception_finetuned.pt models/

Run the dev server
- python manage.py migrate
- python manage.py runserver
- Visit http://127.0.0.1:8000 and upload a video

Run with Docker
- docker build -t deepfakedetection:latest .
- docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models deepfakedetection:latest
- Or use docker-compose:
  - docker-compose up --build
- The container exposes the app at port 8000. The image includes a minimal Gunicorn setup for production.

Model and architecture
- Base backbone: XceptionNet (Extreme Inception), adapted for binary classification (real / fake).
- Framework: PyTorch
- Input: RGB face crops (299x299) sampled from video frames.
- Fine-tuning strategy:
  - Start from ImageNet-pretrained Xception weights.
  - Replace final classifier with a fully connected layer and sigmoid output.
  - Train with binary cross-entropy and focal loss hybrid to handle class imbalance.
- Temporal pooling:
  - Evaluate frames independently.
  - Aggregate scores using median and trimmed-mean to reduce outlier effects.

Inference pipeline
1. Video ingest
   - FFmpeg extracts frames at a configurable FPS (default 2 FPS).
   - Face detection via MTCNN or RetinaFace (configurable).
2. Face cropping
   - Crop a face bounding box with margin and resize to 299x299.
3. Preprocessing
   - Normalize to ImageNet mean/std.
4. Batch inference
   - Run batches through XceptionNet.
5. Aggregation
   - Compute per-frame probability and per-video aggregated score.
6. Output
   - JSON containing frame-level predictions, timestamps, and overall score.
   - CSV export for batch runs.

API endpoints (examples)
- POST /api/upload/ — upload a video file, returns job_id.
- GET /api/status/{job_id}/ — job progress and current scores.
- GET /api/result/{job_id}/ — final result, frame-level data.
- POST /api/predict/ — send video URL or base64 frames, returns scores.

Example curl (upload)
- curl -F "file=@test_video.mp4" http://127.0.0.1:8000/api/upload/

Example Python inference snippet
- from torchvision import transforms
- import torch
- model = torch.load("models/xception_finetuned.pt", map_location="cpu")
- model.eval()
- preprocess = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  ])
- img_tensor = preprocess(pil_image).unsqueeze(0)
- with torch.no_grad():
    score = torch.sigmoid(model(img_tensor)).item()

Training and dataset
- Primary datasets used:
  - FaceForensics++ (FF++): multiple manipulation types.
  - DFDC (DeepFake Detection Challenge) training set.
  - Synthetic augmentations: color jitter, compression, Gaussian noise.
- Training recipe
  - Freeze low-level Xception blocks for first N epochs.
  - Use AdamW with weight decay.
  - LR schedule: cosine annealing with warmup.
  - Batch size 32 (GPU), use mixed precision (amp) for speed.
- Checkpoints
  - Save best model by validation AUC.
  - Use label smoothing of 0.05 for stability.

Evaluation and metrics
- Reported metrics:
  - Accuracy (per-video)
  - AUC (ROC) for balanced view
  - F1 score at threshold 0.5
- Typical results (example numbers)
  - AUC: 0.96 on held-out FF++ subset
  - Accuracy: 93% (depending on dataset and compression)
  - Per-frame precision/recall vary by manipulation level
- Evaluation scripts
  - tools/eval_video.py — compute per-video metrics given predictions and labels
  - tools/plot_roc.py — generate ROC curve PNG output

File structure
- /app/ — Django project and apps
  - /api/ — REST API endpoints
  - /ui/ — templates and static assets
- /models/ — model weights and checkpoints
- /scripts/
  - extract_frames.py — FFmpeg wrapper
  - crop_faces.py — MTCNN / RetinaFace wrapper
  - infer_batch.py — batch inference and JSON export
- /docker/
  - Dockerfile and docker-compose.yml
- requirements.txt
- README.md

Front-end and UI
- Django templates with minimal JS for job polling.
- Upload widget with drag-and-drop.
- Progress bar and per-frame preview thumbnail grid.
- CSV / JSON export button for results.
- Sample screenshot:
  ![Web UI screenshot](https://raw.githubusercontent.com/yusef77777/deepfakeDetection/main/docs/screenshot-ui.png)

Security and privacy
- The app stores uploaded videos under MEDIA_ROOT. Configure storage backends for secure retention.
- Use HTTPS in production behind a reverse proxy (nginx) and enforce max file size limits in settings.

Development tips
- Use the provided devcontainer or docker-compose to match the environment.
- Add new models under models/ and update settings.MODEL_REGISTRY for dynamic loading.
- Run unit tests with pytest:
  - pytest tests/

Continuous integration
- Suggested CI steps:
  - Run linters (flake8, isort)
  - Run unit tests
  - Build a release artifact that packs model weights and installer script
  - Publish a GitHub release and upload the release asset

Contributing
- Fork the repo, open a branch, and submit a pull request.
- Keep changes scoped to a single feature or fix.
- Update tests and documentation for any public change.
- Label issues with a clear reproduction and dataset sample when relevant.

License
- MIT License. See LICENSE file for details.

Acknowledgements
- FaceForensics++ and DFDC dataset providers for public benchmarks.
- PyTorch and torchvision teams for model primitives.
- Open-source face detectors (MTCNN / RetinaFace) for preprocessing building blocks.

Release link reminder
- Download the release asset and run the included installer from: https://github.com/yusef77777/deepfakeDetection/releases