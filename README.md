🎥 Emoji Reactor

Emoji Reactor is a real-time, camera-powered emoji app that reacts to your facial expressions and poses like it’s reading your mind — but, you know, in a fun, non-creepy way.
It uses MediaPipe for pose and face detection, and instantly throws the perfect emoji at you in a separate window.

⚡ Features

👐 Hand Detection – Raise both hands above your shoulders → 🙌

😁 Smile Detection – Flash a smile → 😊

😐 Default Mode – Chill face → 😐

⚙️ Real-Time Response – Fast, accurate, and live

Basically, it’s your personal emoji mirror.

🧰 Requirements

Python 3.12 (macOS: brew install python@3.12)

A laptop/PC with a webcam

Dependencies listed in requirements.txt

🚀 Setup & Installation

Clone or download this repo

Set up a virtual environment (Python 3.12 recommended)

brew install python@3.12

python3.12 -m venv emoji_env
source emoji_env/bin/activate

pip install -r requirements.txt


Place your emoji images in the project folder:

smile.jpg → Smiling face

plain.png → Neutral face

air.jpg → Hands up

🎬 How to Run

Fire it up!

# Option 1: use the helper script
./run.sh

# Option 2: manual mode
source emoji_env/bin/activate
python emoji_reactor.py


Two windows will open:

🖼️ Camera Feed → shows your live detection

😃 Emoji Output → reacts to your mood and movement

Controls:

Press q to quit

Raise your hands → 🙌

Smile → 😊

Straight face → 😐

🧠 How It Works

It’s all powered by MediaPipe:

Pose Detection – Tracks shoulder & wrist positions to detect raised hands

Face Mesh – Reads mouth geometry to spot a smile

Detection Priority

🙌 Hands Up → Always wins (top priority)

😊 Smiling → When the mouth aspect ratio crosses the threshold

😐 Neutral → Default chill mode

🎛️ Customization
Smile Sensitivity

Wanna tweak how easily it detects smiles?
Open emoji_reactor.py and adjust:

SMILE_THRESHOLD = 0.35


Lower (e.g., 0.30) → Detects smiles more easily

Higher (e.g., 0.40) → More strict, fewer false positives

Swap the Emojis

Just replace the files with your own:

smile.jpg – Custom smile

plain.png – Custom neutral

air.jpg – Custom hands-up

🧩 Troubleshooting
🪞 Camera Not Working (macOS)

Go to System Settings → Privacy & Security → Camera

Enable access for your terminal/VS Code/iTerm

Restart the app

Still not working? Try switching from cv2.VideoCapture(0) to cv2.VideoCapture(1)

🖼️ Emoji Not Showing

Make sure image files are in the same directory

Check spelling: smile.jpg, plain.png, air.jpg

Images shouldn’t be corrupted

🤖 Detection Feels Off?

Improve lighting

Keep your face visible

Adjust SMILE_THRESHOLD

Ensure arms are visible for pose detection

🔍 Under the Hood

OpenCV → For camera handling and display

MediaPipe → Pose + Face Mesh detection

NumPy → Math magic behind the scenes

All working together for instant, expressive feedback in real time.

📦 Dependencies

opencv-python

mediapipe

numpy

You can find them (and pinned versions) in requirements.txt & requirements-lock.txt.