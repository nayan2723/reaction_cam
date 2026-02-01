# 🎥 Emoji Reactor

Emoji Reactor is a real-time, camera-powered emoji app that reacts to your facial expressions and poses — like it’s reading your mind, but in a fun, non-creepy way. It uses **MediaPipe** for pose + face detection and displays the correct emoji in a separate output window.

---

## ⚡ Features

* 👐 **Hand Detection** – Raise both hands above your shoulders → 🙌
* 😁 **Smile Detection** – Flash a smile → 😊
* ⚙️ **Real-Time Feedback** – Fast, accurate detection
* 🔄 **Two-Window Display** – Live camera + emoji output

---

## 🧰 Requirements

* Python **3.12**
* Webcam-enabled laptop/PC
* Dependencies listed in `requirements.txt`

---

## 🚀 Setup & Installation

### 1️⃣ Clone the repository

```
git clone <your-repo-url>
cd emoji-reactor
```

### 2️⃣ Create a virtual environment

```
python3.12 -m venv emoji_env
source emoji_env/bin/activate   # macOS/Linux
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Add your emoji images

Place the required emoji files in the project folder:

* **smile.jpg** → Smiling emoji
* **plain.png** → Neutral emoji
* **air.jpg** → Hands-up emoji

---

## 🎬 How to Run

### **Option 1 — Run script**

```
./run.sh
```

### **Option 2 — Manual run**

```
source emoji_env/bin/activate
python emoji_reactor.py
```

### Output

Two windows open:

* 🖼️ **Camera Feed** – Real-time pose/face detection
* 😃 **Emoji Output** – Shows emoji based on your expression/movement

### Controls

* Press **q** to quit
* Raise both hands → 🙌
* Smile → 😊
* Neutral face → 😐

---

## 🧠 How It Works

Emoji Reactor uses MediaPipe for two forms of detection:

### **Pose Detection**

* Tracks **wrist** + **shoulder** positions
* If wrists > shoulders vertically → trigger hands-up 🙌

### **Face Mesh Detection**

* Measures mouth aspect ratio
* Determines whether smile threshold is crossed

### **Detection Priority Order**

1. 🙌 Hands Up
2. 😊 Smile
3. 😐 Neutral

---

## 🎛️ Customization

### Adjust Smile Sensitivity

Inside `emoji_reactor.py`:

```
SMILE_THRESHOLD = 0.35
```

* Lower → detects smiles more easily
* Higher → more strict detection

### Change Emojis

Replace these image files:

* `smile.jpg`
* `plain.png`
* `air.jpg`

---

## 🧩 Troubleshooting

### 🪞 Camera Issues (macOS)

* Go to **System Settings → Privacy & Security → Camera**
* Enable access for Terminal/VS Code
* Restart the app
* If needed, switch camera index:

```
cv2.VideoCapture(1)
```

### 🖼️ Emoji Missing

* Ensure image files exist
* Verify filenames
* Check image format/corruption

### 🤖 Detection Off

* Improve lighting
* Adjust SMILE_THRESHOLD
* Ensure hands/face are in frame

---

## 🔍 Tech Behind the Scenes

* **OpenCV** → Captures + displays video
* **MediaPipe** → Pose + face mesh detection
* **NumPy** → Mathematical calculations
* **Custom Logic** → Thresholds + detection rules

---

## 📦 Dependencies

```
opencv-python
mediapipe
numpy
```

Dependencies are pinned in `requirements.txt` and `requirements-lock.txt`.

---
