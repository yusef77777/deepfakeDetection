# DeepDeception 🎭🧠

A Deep Learning-based web application for **deepfake video detection**, powered by the **Xception** model and deployed via **Railway**.  
Developed as part of our Final Year Project at Bahria University.

---

## 🚀 Overview

**DeepDeception** uses a fine-tuned **Xception (Extreme Inception)** model to detect deepfakes from video uploads. The system is integrated into a **Django**-based web app with a clean user interface, a feedback submission form, and a backend database to store user feedback and email addresses.

---

## 🧠 Core Features

- 🎯 **Xception-based Deepfake Detection**  
  Trained and tuned on real vs fake video data (DFDC Dataset)

- 🖥️ **Django Web App Interface**  
  Upload video files and get instant predictions

- 📨 **Feedback System**  
  Users can submit feedback and emails, which are stored in a secure database

- ☁️ **Deployed on Railway**  
  Accessible online with scalable backend hosting

---

## ⚙️ Tech Stack

| Component        | Technology        |
|------------------|------------------|
| Deep Learning    | Xception (Keras/TensorFlow) |
| Web Framework    | Django (Python)  |
| Frontend         | HTML, CSS, Bootstrap |
| Database         | SQLite / PostgreSQL (Railway) |
| Deployment       | Railway          |
| Video Handling   | OpenCV, FFmpeg   |

---

## 📂 Project Structure
DeepDeception/
│
├── model/ # Xception model + inference code
├── webapp/ # Django project
│ ├── templates/
│ ├── static/
│ ├── views.py
│ ├── urls.py
│ └── models.py # Stores email + feedback
│
├── media/ # Uploaded videos (temp storage)
├── requirements.txt
└── README.md


---

## 💡 How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/YourUsername/DeepDeception.git
cd DeepDeception

pip install -r requirements.txt

python manage.py runserver
📬 Feedback System
Users can submit feedback through the web interface.

Submitted feedback and email addresses are stored securely in the backend database.

Admins can view and manage feedback entries via the Django admin panel (if enabled).

🌐 Live Demo
🟢 Hosted on Railway
URL: [Insert your deployed Railway app link here]

📊 Model Performance
Architecture: Xception (pretrained, fine-tuned)

Dataset: Deepfake Detection Challenge (DFDC)

Accuracy: ~94% on validation set

Input: Short video clips (MP4)

Output: Real / Fake classification with confidence score

🧑‍💻 Authors
Abdulrehman Qureshi
GitHub
AI Researcher | Computer Vision & Web Systems

Maheen Sheikh
GitHub
Full-stack Developer | AI Enthusiast

🎓 Academic Info
Project: Final Year Project – DeepDeception

Institute: Bahria University

Supervisor: Dr. Muhammad Usman Hashmi



