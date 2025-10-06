#!/usr/bin/env python3
"""
Emoji Reactor - A real-time camera-based emoji display application
Uses MediaPipe for pose detection (hands up) and face mesh detection (smiling)
Displays different emojis based on your actions and expressions.
"""

import cv2
import mediapipe as mp
import numpy as np

# --- SETUP AND INITIALIZATION ---

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION CONSTANTS ---
SMILE_THRESHOLD = 0.35  # Adjust this value based on your smile sensitivity
# MacBook Pro screen is typically 1440x900 or 1680x1050, so half would be around 720x450 or 840x525
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# --- LOAD AND PREPARE EMOJI IMAGES ---
try:
    # Load images from files (using your specific filenames)
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")

    # Check if images loaded successfully
    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")

    # Resize emojis to a consistent size
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    
    print("✅ All emoji images loaded successfully!")
    
except Exception as e:
    print("❌ Error loading emoji images! Make sure they are in the correct folder and named properly.")
    print(f"Error details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (smiling face)")
    print("- plain.png (straight face)")
    print("- air.jpg (hands up)")
    exit()

# Create a blank image for cases where an emoji is missing
blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# --- MAIN LOGIC ---

# Start webcam capture
print("🎥 Starting webcam capture...")
cap = cv2.VideoCapture(0)

# Check if webcam is available
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
    exit()

# Initialize named windows with specific sizes
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

# Set window sizes and positions
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)

# Position windows side by side
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("🚀 Starting emoji detection...")
print("📋 Controls:")
print("   - Press 'q' to quit")
print("   - Raise your hands above your shoulders for hands up emoji")
print("   - Smile for smiling emoji")
print("   - Keep a straight face for straight face emoji")

# Instantiate MediaPipe models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️  Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror-like display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, mark the image as not writeable
        image_rgb.flags.writeable = False

        # --- DETECTION LOGIC ---
        
        # Default state is a straight face
        current_state = "STRAIGHT_FACE"

        # 1. Process for Pose (Hands Up) - This has the highest priority
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            
            # Get coordinates for shoulders and wrists
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Check if either wrist is above its corresponding shoulder
            # MediaPipe coordinates: y=0 at top, y=1 at bottom
            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"
        
        # 2. Process for Facial Expression (if hands are not up)
        if current_state != "HANDS_UP":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Mouth corner landmarks
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    # Upper and lower lip landmarks
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]

                    # Calculate mouth aspect ratio to detect a smile
                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                    
                    if mouth_width > 0:  # Avoid division by zero
                        mouth_aspect_ratio = mouth_height / mouth_width
                        if mouth_aspect_ratio > SMILE_THRESHOLD:
                            current_state = "SMILING"
                        else:
                            current_state = "STRAIGHT_FACE"
        
        # --- DISPLAY LOGIC ---
        
        # Select the emoji to display based on the detected state
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        # Resize camera frame to match window size
        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Add the status text to the main camera feed
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add instructions text
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the camera feed and emoji
        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("✅ Application closed successfully!")
