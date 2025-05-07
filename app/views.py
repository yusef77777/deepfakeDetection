import os
import gc
import json
import cv2
import numpy as np
import shutil
import requests
from PIL import Image
import imagehash

# The most aggressive GPU disabling possible - must be before ANY TensorFlow imports
# This completely prevents TensorFlow from seeing or trying to use any GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_ALLOCATOR"] = "null"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# Railway-specific memory limits
os.environ["MALLOC_ARENA_MAX"] = "2"  # Limit memory arenas

# Now import TensorFlow with a modified import system that ignores GPU packages
import sys
original_import = __import__

def import_hook(name, *args, **kwargs):
    if name.startswith('tensorflow.python.eager.') and 'gpu' in name:
        # Skip importing GPU-related modules
        return sys.modules.get(name, None)
    return original_import(name, *args, **kwargs)

sys.meta_path = [
    importer for importer in sys.meta_path
    if not hasattr(importer, 'find_spec') or 'tensorflow.python.eager.memory_tests.ops_test' not in str(importer.find_spec)
]

# Now safe to import TensorFlow
import tensorflow as tf

# Explicitly configure TF to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Disable JIT compilation which might try to use GPU
tf.config.optimizer.set_jit(False)

# Further restrict TF to use minimal resources
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
    except RuntimeError as e:
        print(e)

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

# Model settings
MODEL_URL = 'https://huggingface.co/abdulrehman77/deepfakedetection/resolve/main/XSoftmax-%201st%20high%20P.h5'
MODEL_PATH = '/tmp/XSoftmax-1st-highP.h5'

# Load Haar cascade face detector once
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize model globally
model = None
has_warned = False  # Flag to track if we've shown model loading warning

# Railway-friendly constants
MAX_FRAMES_TO_PROCESS = 10  # Limit total processed frames
MAX_FACES_TO_EXTRACT = 8   # Limit face extraction
FRAME_SKIP_RATE = 45        # Skip more frames to reduce processing

def initialize_model():
    """Download and load the model if needed"""
    global model, has_warned
    
    # Show warning only once
    if not has_warned:
        print("===============================================")
        print("IMPORTANT: Loading model in CPU-only mode")
        print("Any GPU-related warnings should be ignored")
        print("===============================================")
        has_warned = True
    
    # Download the model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("Downloading the model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    
    # Load the model with memory optimization
    if model is None:
        print("Loading the model...")
        try:
            # The most explicit way to force CPU usage
            with tf.device('/CPU:0'):
                # Load with minimal TF options
                model = tf.keras.models.load_model(
                    MODEL_PATH, 
                    compile=False,
                    options=tf.saved_model.LoadOptions(
                        experimental_io_device='/job:localhost'
                    )
                )
            print("Model loaded successfully in CPU-only mode!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try again with memory optimization
            clear_resources()
            with tf.device('/CPU:0'):
                model = tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False
                )
            print("Model loaded with fallback method!")

def clear_resources():
    """Clear TensorFlow session to free memory"""
    gc.collect()
    
    # Additional memory cleanup
    import ctypes
    if hasattr(ctypes, 'windll'):  # Windows-specific
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except:
            pass


@tf.function
def predict_image(img_array):
    return model(img_array, training=False)

def check_faces_in_video(video_path):
    """Check if there are any faces in the video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return False

    # Process frames at intervals to speed up checking
    frame_skip = FRAME_SKIP_RATE
    frame_count = 0
    max_frames_to_check = 100  # Limit how many frames we check
    
    frames_checked = 0
    
    while frames_checked < max_frames_to_check:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
            
            frames_checked += 1
            
            if len(faces) > 0:  # If faces are detected
                cap.release()
                return True
        
        frame_count += 1
    
    cap.release()
    return False

def capture_video_frames(path):
    """Extract frames with faces from video"""
    output_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
    frame_skip = FRAME_SKIP_RATE  # Increased skip rate to reduce processing
    min_face_size = 60

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
    
    # Maximum faces to extract (limit processing for Railway)
    max_faces = MAX_FACES_TO_EXTRACT
    
    frame_count = 0
    saved_faces = 0
    
    print("Extracting faces from video...")
    
    # For long videos, increase skip rate further
    if total_frames > 1000:
        frame_skip = 60
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Resize frame to reduce memory usage
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
            # Skip frames that are too dark overall
            if frame.mean() < 30:  # Skip very dark frames
                frame_count += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(min_face_size, min_face_size)
            )

            for i, (x, y, w, h) in enumerate(faces):
                # Skip faces with unusual aspect ratios
                aspect_ratio = w / float(h)
                if not (0.75 <= aspect_ratio <= 1.33):
                    continue

                # Extract and check face
                face = frame[y:y+h, x:x+w]
                
                # Skip low brightness faces
                brightness = face.mean()
                if brightness < 40:
                    continue
                    
                # Skip small faces
                if w < min_face_size or h < min_face_size:
                    continue

                # Save the face
                filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                cv2.imwrite(filename, face)
                saved_faces += 1
                
                # Progress indicator
                if saved_faces % 5 == 0:
                    print(f"Saved {saved_faces} faces...", end="\r")
                
                # Limit number of saved faces
                if saved_faces >= max_faces:
                    cap.release()
                    print(f"\nReached limit: {saved_faces} face crops saved.")
                    remove_non_face_and_duplicate_frames(output_dir)
                    return

        frame_count += 1
        
        # Safety limit to prevent memory issues with very long videos
        if frame_count > 5000:
            break

    cap.release()
    print(f"\nDone: {saved_faces} face crops saved.")

    # Remove non-face and duplicate frames
    remove_non_face_and_duplicate_frames(output_dir)

def remove_non_face_and_duplicate_frames(directory):
    """Remove duplicate and non-face images"""
    seen_hashes = set()
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for file in files:
        file_path = os.path.join(directory, file)
        try:
            image = Image.open(file_path)
            image_hash = imagehash.average_hash(image)

            img_cv = cv2.imread(file_path)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                os.remove(file_path)
            elif str(image_hash) in seen_hashes:
                os.remove(file_path)
            else:
                seen_hashes.add(str(image_hash))

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            # Remove problematic files
            if os.path.exists(file_path):
                os.remove(file_path)

def get_confidence_label(prediction):
    """Format prediction label"""
    return "Real ✅" if prediction == "Real" else "Fake ❌"

def evaluate_frames(directory):
    """Evaluate extracted frames to determine if video is deepfake"""
    # Ensure model is loaded
    initialize_model()
    
    total_confidence = 0
    num_frames = 0
    results = []
    fake_series = []
    real_series = []
    frame_labels = []
    fake_count = 0
    real_count = 0
    confidence_series = []
    
    # Get all image paths and limit the number to prevent memory issues
    all_files = sorted([f for f in os.listdir(directory) if f.endswith((".jpg", ".png"))])
    
    # If we have too many frames, select evenly distributed frames
    if len(all_files) > MAX_FRAMES_TO_PROCESS:
        # Select evenly distributed frames
        step = max(1, len(all_files) // MAX_FRAMES_TO_PROCESS)
        selected_files = [all_files[i] for i in range(0, len(all_files), step)][:MAX_FRAMES_TO_PROCESS]
    else:
        selected_files = all_files
    
    image_paths = [os.path.join(directory, f) for f in selected_files]
    filenames = selected_files
    
    # If no frames found
    if not image_paths:
        return [], 0, "No frames found.", 0, 0, "N/A", [], [], [], []
    
    # Process one image at a time to minimize memory usage
    total_images = len(image_paths)
    print(f"\nProcessing {total_images} images:")
    
    for i, (img_path, filename) in enumerate(zip(image_paths, filenames)):
        try:
            # Clear previous tensors to free memory
            

            gc.collect()
            
            # Print progress
            progress = ((i+1) / total_images) * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{total_images})", end="\r")
            
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype(np.float32))
            
            # Force CPU prediction
            with tf.device('/CPU:0'):
                prediction = predict_image(img_array)[0].numpy()

            
            confidence = float(prediction[1])
            confidence_series.append(round(confidence, 4))
            total_confidence += confidence
            num_frames += 1
            
            # Label for each frame
            frame_labels.append(f"Frame {i+1}")
            
            if confidence >= 0.5:
                results.append((filename, "Fake", confidence))
                fake_count += 1
                fake_series.append(1)
                real_series.append(0)
            else:
                results.append((filename, "Real", confidence))
                real_count += 1
                fake_series.append(0)
                real_series.append(1)
                
        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")
            # Continue with next image
            continue
    
    print("\nProcessing complete!                ")
    
    # Calculate overall prediction
    if num_frames > 0:
        average_confidence = total_confidence / num_frames
        display_confidence = round(average_confidence * 100, 2)
        prediction_type = "Fake" if average_confidence >= 0.5 else "Real"
        overall_prediction = "The video is predicted as a deepfake." if prediction_type == "Fake" else "The video is predicted as real."
        overall_label = get_confidence_label(prediction_type)
    else:
        average_confidence = 0
        display_confidence = 0
        overall_prediction = "No frames found."
        overall_label = "N/A"

    # Clear resources after processing
    clear_resources()
    
    return results, display_confidence, overall_prediction, real_count, fake_count, overall_label, fake_series, real_series, frame_labels, confidence_series

def clean_media_directory():
    """Clean up media directory to free space"""
    media_dir = settings.MEDIA_ROOT
    
    if os.path.exists(media_dir):
        # Delete the media directory and all its contents
        shutil.rmtree(media_dir)
    
    # Create the media directory again
    os.makedirs(media_dir)

def upload_video(request):
    """Handle video upload and processing"""
    # Initialize model when view is first accessed
    initialize_model()
    
    # Clean media directory
    clean_media_directory()

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            fs = FileSystemStorage()
            video_path = fs.save(video.name, video)
            video_full_path = os.path.join(settings.MEDIA_ROOT, video_path)

            # Check if faces are detected in the video
            faces_detected = check_faces_in_video(video_full_path)
            if not faces_detected:
                # If no faces are detected, stop further processing and send message to frontend
                return render(request, 'landing_page.html', {'form': form, 'error_message': 'No faces detected in the video.'})

            # Proceed with further processing (frame capture, evaluation, etc.)
            capture_video_frames(video_full_path)
            frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
            
            # Always clear GPU memory before heavy processing
            clear_resources()
            
            results, display_confidence, overall_prediction, real_count, fake_count, overall_label, fake_series, real_series, frame_labels, confidence_series = evaluate_frames(frames_dir)
        
            # Convert all series to native Python types
            confidence_series = [float(val) for val in confidence_series]
            real_series = [int(val) for val in real_series]
            fake_series = [int(val) for val in fake_series]
            frame_labels = [str(label) for label in frame_labels]
            
            # Delete video file after processing to free space
            if os.path.exists(video_full_path):
                try:
                    os.remove(video_full_path)
                except:
                    pass

            return render(request, 'results.html', {
                'results': results,
                'average_confidence': display_confidence,
                'overall_prediction': overall_prediction,
                'real_count': real_count,
                'fake_count': fake_count,
                'overall_label': overall_label,
                'confidence_series': json.dumps(confidence_series),
                'real_series': json.dumps(real_series),
                'fake_series': json.dumps(fake_series),
                'frame_labels': json.dumps(frame_labels),
            })
    else:
        form = VideoUploadForm()

    return render(request, 'landing_page.html', {'form': form})

@csrf_exempt  # Disable CSRF protection for this view temporarily
def email_submission(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')

            if email:
                # Save the email to the database
                Email.objects.create(email=email)
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'error', 'message': 'No email provided'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@csrf_exempt
@require_POST
def submit_feedback(request):
    try:
        data = json.loads(request.body)
        feedback = Feedback(
            rating=data.get('rating'),
            comment=data.get('comment', ''),
            page_url=data.get('page_url', ''),
            ip_address=request.META.get('REMOTE_ADDR')
        )
        feedback.save()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
