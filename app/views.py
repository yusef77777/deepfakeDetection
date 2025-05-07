import os

# More aggressive GPU disabling - must be done before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
import gc
import json
import cv2
import numpy as np
import shutil
import requests
from PIL import Image
import imagehash


# Now import TensorFlow after environment variables are set
import tensorflow as tf

# Make absolutely sure no GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Disable all GPUs
        for device in physical_devices:
            tf.config.set_visible_devices([], 'GPU')
            print(f"Disabled GPU: {device}")
    except RuntimeError as e:
        print(f"Error disabling GPU: {e}")

# Further CPU configuration
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

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

def initialize_model():
    """Download and load the model if needed"""
    global model
    
    # Download the model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("Downloading the model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    
    # Load the model
    if model is None:
        print("Loading the model...")
        # Force CPU usage with a custom device scope
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")

def clear_resources():
    """Clear TensorFlow session to free memory"""
    tf.keras.backend.clear_session()
    gc.collect()

def check_faces_in_video(video_path):
    """Check if there are any faces in the video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return False

    # Process frames at intervals to speed up checking
    frame_skip = 30
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
            
            if len(faces) > 0:  # If faces are detected
                cap.release()
                return True
        
        frame_count += 1
    
    cap.release()
    return False

def capture_video_frames(path):
    """Extract frames with faces from video"""
    output_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
    frame_skip = 20
    min_face_size = 60

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_faces = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adaptive frame skip rate based on video length
    if total_frames > 1000:
        frame_skip = 30
    elif total_frames > 500:
        frame_skip = 20
    else:
        frame_skip = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(min_face_size, min_face_size)
            )

            for i, (x, y, w, h) in enumerate(faces):
                # Filter faces by aspect ratio and brightness
                aspect_ratio = w / float(h)
                if not (0.75 <= aspect_ratio <= 1.33):
                    continue

                face = frame[y:y+h, x:x+w]
                brightness = face.mean()
                if brightness < 40:
                    continue

                filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                cv2.imwrite(filename, face)
                saved_faces += 1
                
                # Limit number of saved faces to prevent excessive processing
                if saved_faces >= 30:
                    cap.release()
                    print(f"\nDone: {saved_faces} face crops saved (limit reached).")
                    remove_non_face_and_duplicate_frames(output_dir)
                    return

        frame_count += 1

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
    
    # Process all images in batches for efficiency
    image_paths = []
    filenames = []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            image_paths.append(img_path)
            filenames.append(filename)
    
    # If no frames found
    if not image_paths:
        return [], 0, "No frames found.", 0, 0, "N/A", [], [], [], []
    
    # Reduce batch size to prevent memory issues
    batch_size = 4  # Decreased from 8 to reduce memory usage
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_names = filenames[i:i+batch_size]
        batch_arrays = []
        
        for img_path in batch_paths:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype(np.float32))
            batch_arrays.append(img_array[0])
        
        # Stack arrays for batch prediction
        batch_input = np.stack(batch_arrays)
        
        # Force CPU prediction with smaller memory footprint
        with tf.device('/CPU:0'):
            batch_predictions = model.predict(batch_input, verbose=0)
        
        for j, (filename, prediction) in enumerate(zip(batch_names, batch_predictions)):
            confidence = float(prediction[1])
            confidence_series.append(round(confidence, 4))
            total_confidence += confidence
            num_frames += 1
            
            # Label for each frame
            frame_labels.append(f"Frame {i+j+1}")
            
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
            results, display_confidence, overall_prediction, real_count, fake_count, overall_label, fake_series, real_series, frame_labels, confidence_series = evaluate_frames(frames_dir)
        
            # Convert all series to native Python types
            confidence_series = [float(val) for val in confidence_series]
            real_series = [int(val) for val in real_series]
            fake_series = [int(val) for val in fake_series]
            frame_labels = [str(label) for label in frame_labels]

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
    




    
