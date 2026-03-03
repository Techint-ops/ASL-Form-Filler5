# ASL Form Filler 4 🖐️📄
**Online Multimodal Form Filling Application with Sign Language Interpreter**

## 📌 Project Overview
ASL Form Filler 4 is an accessibility-focused application that helps users fill structured forms using **American Sign Language (ASL)** gestures.

The system captures real-time webcam input, detects hand landmarks using **MediaPipe**, classifies ASL gestures into characters (**A-Z, 0-9**), and inserts the recognized characters into form fields.

This project was developed as part of a **Study Project**, with the goal of exploring how computer vision + machine learning can support accessible input methods.

---

## 🎯 Core Features
- Real-time webcam capture using OpenCV  
- 21 hand landmark detection using MediaPipe  
- ASL character recognition using a trained Random Forest classifier  
- Separate classification modes:
  - Alphabet detection (A-Z)
  - Number detection (0-9)
- Hold-to-confirm mechanism (40-frame threshold) to prevent accidental typing  
- Structured form interface (First Name, Last Name, Age, etc.)  
- Keyboard-based form navigation  
- Local saving of filled form data into 'form_output.txt' 
- Side-by-side camera feed and form panel display  

---

## 🧠 System Architecture (High Level)
Webcam → MediaPipe Hand Landmarks → Feature Extraction → Random Forest Classifier → Character Prediction → Form Field Insertion → Local File Storage

---

## 🛠 Technology Stack
- Python 3.11
- OpenCV
- MediaPipe
- Scikit-learn (Random Forest)
- NumPy
- Matplotlib
- Autocorrect
- ImageIO

---

## 🚀 Installation Guide (Windows)

### 1) Clone the Repository
git clone https://github.com/Techint-ops/ASL-Form-Filler4.git  
cd ASL-Form-Filler4  

### 2) Create and Activate Virtual Environment (Recommended)
py -3.11 -m venv venv  
venv\Scripts\activate  

Check Python:
python --version

### 3) Install Dependencies
pip install -r requirements.txt

### 4) Run the Application
python main.py

After running, the webcam window and the form panel should open.

---

## ⌨️ Controls & Usage Instructions
- R  → Start gesture detection
- M  → Toggle between Letters mode and Numbers mode
- Tab → Move to next form field
- Q  → Move to previous form field
- Backspace → Delete last character
- Enter → Save form to 'form_output.txt'
- Esc → Exit application

---

## 🧪 Training Pipeline (Summary)
The ASL classifier was trained using a custom dataset captured via webcam.
- ~300 images per letter (A-Z)
- ~300 images per number (0-9)

Model used:
- Random Forest classifier
- Separate models for letters and numbers
- Training notebook: 'train.ipynb'

To retrain (optional):
jupyter notebook train.ipynb

---

## 📊 Performance Notes
- Runs in real-time (CPU based)
- Most reliable in stable lighting
- Recommended webcam resolution: minimum 720p
- Accuracy depends on lighting, background clutter, and gesture stability

---

## ⚠️ Current Limitations
- Supports only static ASL characters (not word-level dynamic gestures)
- Single-hand tracking only
- No cloud/database storage (only local file save)
- Limited robustness in low light
- Manual toggle required for Letters vs Numbers mode

---

## 🔮 Future Enhancements
- Word-level ASL recognition (dynamic gestures)
- Multi-hand + facial expression integration
- Cloud-based form submission
- PDF/CSV/JSON export
- Mobile deployment
- Context-aware autocorrect
- Support for other sign languages (e.g., ISL, BSL)

---

## 🎥 Demonstration Video
https://youtu.be/Wd6g0Gqvxho

---

## 👥 Team Techint
- Aniketh Chittiprolu
- Varun Shivan
- S. Trishaa
- Yashvi Shah

Supervisor: Prof. Dharmateja Adapa

---

## 📄 License
Developed for academic purposes as part of a Study Project.

---

## ⭐ Acknowledgements
- Google MediaPipe
- OpenCV
- Scikit-learn
- Kaggle ASL datasets
- Open-source community resources

