
import os
import shutil
import gc
import json
import cv2
import numpy as np
from PIL import Image
import imagehash
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.xception import preprocess_input
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import Feedback
from .forms import VideoUploadForm
from .models import Email



def clear_gpu_memory():
    ops.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()
clear_gpu_memory()





# Load the model once when the server starts
model = tf.keras.models.load_model(r"C:\Users\creat\Desktop\semesters\7th semester\deepfake_fyp1\models\best models\XSoftmax- 1st high P.h5", compile=False)





def check_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        
        if len(faces) > 0:  # If faces are detected
            cap.release()
            return True
    
    cap.release()
    return False


# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def FrameCapture(path):
    clear_gpu_memory()
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
    clear_gpu_memory() 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(min_face_size, min_face_size)
            )

            for i, (x, y, w, h) in enumerate(faces):
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.75 or aspect_ratio > 1.33:
                    continue

                face = frame[y:y+h, x:x+w]
                brightness = face.mean()
                if brightness < 40:
                    continue

                filename = os.path.join(output_dir, f"face_{frame_count}_{i}.jpg")
                cv2.imwrite(filename, face)

                print(f"[✓] Saved {filename}")
                saved_faces += 1

        frame_count += 1

    cap.release()
    print(f"\nDone: {saved_faces} face crops saved.")

    # Optional: Remove non-face and duplicate frames
    remove_non_face_and_duplicate_frames(output_dir)


def remove_non_face_and_duplicate_frames(directory):
    
    seen_hashes = set()
    files = os.listdir(directory)

    for file in files:
        file_path = os.path.join(directory, file)

        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                image = Image.open(file_path)
                image_hash = imagehash.average_hash(image)

                img_cv = cv2.imread(file_path)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    print(f"No faces detected, removing: {file}")
                    os.remove(file_path)
                elif image_hash in seen_hashes:
                    print(f"Duplicate found and removing: {file}")
                    os.remove(file_path)
                else:
                    seen_hashes.add(image_hash)

            except Exception as e:
                print(f"Error processing file {file}: {e}")




def get_confidence_label(confidence, prediction):
    if prediction == "Real":
        return "Real ✅"
    else:
        return "Fake ❌"


def evaluate_frames(directory):
    clear_gpu_memory()
    total_confidence = 0
    num_frames = 0
    results = []
    fake_series = []
    real_series = []
    frame_labels = []
    fake_count = 0
    real_count = 0
    confidence_series = []
    



    for idx, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype(np.float32))  # Use Xception preprocessing

            prediction = model.predict(img_array, verbose=0)[0]
            confidence = prediction[1]
            confidence_series.append(round(confidence, 4))
            total_confidence += confidence
            num_frames += 1

            # Label for each frame (e.g., "Frame 1", "Frame 2", ...)
            frame_labels.append(f"Frame {idx + 1}")

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

    if num_frames > 0:
        average_confidence = total_confidence / num_frames
        display_confidence = round(average_confidence * 100, 2)
        prediction_type = "Fake" if average_confidence >= 0.5 else "Real"
        overall_prediction = "The video is predicted as a deepfake." if prediction_type == "Fake" else "The video is predicted as real."
        overall_label = get_confidence_label(average_confidence, prediction_type)
    else:
        average_confidence = 0
        display_confidence = 0
        overall_prediction = "No frames found."
        overall_label = "N/A"

    return results, display_confidence, overall_prediction, real_count, fake_count, overall_label, fake_series, real_series, frame_labels, confidence_series





def upload_video(request):
    clear_gpu_memory()

    media_dir = settings.MEDIA_ROOT

    # Check if the media directory exists
    if os.path.exists(media_dir):
        # Delete the media directory and all its contents
        shutil.rmtree(media_dir)

    # Create the media directory again
    os.makedirs(media_dir)

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
            FrameCapture(video_full_path)
            frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
            results, display_confidence, overall_prediction, real_count, fake_count, overall_label, fake_series, real_series, frame_labels, confidence_series = evaluate_frames(frames_dir)
            clear_gpu_memory()
            # Convert NumPy types to Python native types
            confidence_series = [float(val) for val in confidence_series]
            real_series = [float(val) for val in real_series]
            fake_series = [float(val) for val in fake_series]
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
    




    