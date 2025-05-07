import os
import gc
import psutil
import json
import cv2
import numpy as np
import shutil
from PIL import Image
import imagehash
from huggingface_hub import hf_hub_download

# CPU-only TensorFlow config for low RAM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_ALLOCATOR"] = "null"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["MALLOC_ARENA_MAX"] = "2"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.optimizer.set_jit(False)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import Feedback, Email
from .forms import VideoUploadForm

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def clear_memory():
    gc.collect()
    try:
        if os.name == 'posix':
            pid = os.getpid()
            with open(f"/proc/{pid}/clear_refs", "w") as f:
                f.write("1")
    except Exception:
        pass
    try:
        process = psutil.Process(os.getpid())
        _ = process.memory_info()
    except Exception:
        pass

def load_model_from_hf():
    model_path = hf_hub_download(
        repo_id="abdulrehman77/deepfakedetection",
        filename="XSoftmax- 1st high P.h5",
        cache_dir=os.path.join(settings.BASE_DIR, 'models')
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model_from_hf()

def check_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
        if len(faces) > 0:
            cap.release()
            return True
    cap.release()
    return False

def FrameCapture(path):
    clear_memory()
    output_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_faces = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 45 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
            for i, (x, y, w, h) in enumerate(faces):
                if not (0.75 <= w / float(h) <= 1.33):
                    continue
                face = frame[y:y+h, x:x+w]
                if face.mean() < 40:
                    continue
                filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                cv2.imwrite(filename, face)
                saved_faces += 1
        frame_count += 1
    cap.release()
    clear_memory()
    remove_non_face_and_duplicate_frames(output_dir)

def remove_non_face_and_duplicate_frames(directory):
    clear_memory()
    seen_hashes = set()
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_obj = Image.open(file_path)
                image_hash = imagehash.average_hash(image_obj)
                img_cv = cv2.imread(file_path)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(faces) == 0 or image_hash in seen_hashes:
                    os.remove(file_path)
                else:
                    seen_hashes.add(image_hash)
            except Exception:
                pass
    clear_memory()

def get_confidence_label(confidence, prediction):
    return "Real ✅" if prediction == "Real" else "Fake ❌"

def evaluate_frames(directory):
    clear_memory()
    total_confidence = 0
    num_frames = 0
    results = []
    fake_count = 0
    real_count = 0
    frame_labels = []
    confidence_series = []

    for idx, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype(np.float32))

            prediction = model.predict(img_array, verbose=0)[0]
            confidence = prediction[1]
            confidence_series.append(round(confidence, 4))
            total_confidence += confidence
            num_frames += 1
            frame_labels.append(f"Frame {idx + 1}")

            if confidence >= 0.5:
                results.append((filename, "Fake", confidence))
                fake_count += 1
            else:
                results.append((filename, "Real", confidence))
                real_count += 1

            del img, img_array, prediction
            clear_memory()

    if num_frames > 0:
        avg_conf = total_confidence / num_frames
        label = "Fake" if avg_conf >= 0.5 else "Real"
        return results, round(avg_conf * 100, 2), \
            "The video is predicted as a deepfake." if label == "Fake" else "The video is predicted as real.", \
            real_count, fake_count, get_confidence_label(avg_conf, label), frame_labels, confidence_series
    else:
        return [], 0, "No frames found.", 0, 0, "N/A", [], []

def upload_video(request):
    media_dir = settings.MEDIA_ROOT
    if os.path.exists(media_dir):
        shutil.rmtree(media_dir)
    os.makedirs(media_dir)

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            fs = FileSystemStorage()
            video_path = fs.save(video.name, video)
            full_path = os.path.join(settings.MEDIA_ROOT, video_path)

            if not check_faces_in_video(full_path):
                return render(request, 'landing_page.html', {
                    'form': form,
                    'error_message': 'No faces detected in the video.'
                })

            FrameCapture(full_path)
            frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
            results, avg_conf, prediction_text, real_count, fake_count, label, frame_labels, conf_series = evaluate_frames(frames_dir)

            return render(request, 'results.html', {
                'results': results,
                'average_confidence': avg_conf,
                'overall_prediction': prediction_text,
                'real_count': real_count,
                'fake_count': fake_count,
                'overall_label': label,
                'confidence_series': json.dumps([float(x) for x in conf_series]),
                'frame_labels': json.dumps(frame_labels),
            })
    else:
        form = VideoUploadForm()
    return render(request, 'landing_page.html', {'form': form})

@csrf_exempt
def email_submission(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            if email:
                Email.objects.create(email=email)
                return JsonResponse({'status': 'success'})
            return JsonResponse({'status': 'error', 'message': 'No email provided'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@csrf_exempt
@require_POST
def submit_feedback(request):
    try:
        data = json.loads(request.body)
        Feedback.objects.create(
            rating=data.get('rating'),
            comment=data.get('comment', ''),
            page_url=data.get('page_url', ''),
            ip_address=request.META.get('REMOTE_ADDR')
        )
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
