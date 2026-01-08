#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose, facial expression, and hand gesture detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, deque
from typing import Deque, List, Tuple

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_THRESHOLD = 0.40  # Mouth aspect ratio for smile/laugh
SURPRISED_MOUTH_THRESHOLD = 0.55  # Mouth wide open
SURPRISED_EYE_THRESHOLD = 0.30  # Eyes open wide
ANGRY_EYEBROW_THRESHOLD = 0.015  # Eyebrows lowered
ANGRY_EYE_THRESHOLD = 0.18  # Eyes squinting
SAD_EYE_THRESHOLD = 0.22  # Eyes half-closed
SAD_MOUTH_THRESHOLD = 0.008  # Mouth corners down
CRAZY_HEAD_TILT_THRESHOLD = 12  # Degrees for head tilt
CRAZY_EYE_DIFF_THRESHOLD = 0.12  # Difference between eyes for crazy
HEART_DISTANCE_THRESHOLD = 0.12  # Distance threshold for heart gesture (increased for better detection)
CHIN_HAND_DISTANCE_THRESHOLD = 0.15  # Distance for hand near chin (thinking)
HAND_ABOVE_SHOULDER_OFFSET = 0.05  # Offset for hand above shoulder detection

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# State machine configuration
current_emotion = "PLAIN"
last_emotion = "PLAIN"
STATE_BUFFER_SIZE = 7  # number of recent frames to smooth over
state_buffer: Deque[str] = deque(maxlen=STATE_BUFFER_SIZE)

# Load emoji images
def load_emoji(filename: str, required: bool = False):
    """Load and resize an emoji image."""
    emoji = cv2.imread(filename)
    if emoji is None:
        if required:
            raise FileNotFoundError(f"{filename} not found")
        return None
    return cv2.resize(emoji, EMOJI_WINDOW_SIZE)

try:
    # Load all emoji images - map to actual file names that ship with the repo
    emoji_images = {
        "smile": load_emoji("smile.jpg", required=False),
        "surprised": load_emoji("surprised.jpg", required=False),
        "sad": load_emoji("sad.jpg", required=False),
        "dance": load_emoji("dance.jpg", required=False),
        "air": load_emoji("air.jpg", required=False),
        "heart": load_emoji("heart.jpg", required=False),
        "cat": load_emoji("cat.jpg", required=False),
        "plain": load_emoji("plain.png", required=False),
        "crazy": load_emoji("crazy.jpg", required=False),
        "thinking": load_emoji("thinking.jpg", required=False),
        "middle": load_emoji("middle.jpg", required=False),
        "wave": load_emoji("wave.jpg", required=False),
        "wink": load_emoji("wink.jpg", required=False),
    }

    # Set fallbacks between similar emotions
    # Crazy: prefer crazy.jpg, then wave, then wink, then smile/surprised
    if emoji_images["crazy"] is None:
        if emoji_images["wave"] is not None:
            emoji_images["crazy"] = emoji_images["wave"]
        elif emoji_images["wink"] is not None:
            emoji_images["crazy"] = emoji_images["wink"]
        elif emoji_images["surprised"] is not None:
            emoji_images["crazy"] = emoji_images["surprised"]
        elif emoji_images["smile"] is not None:
            emoji_images["crazy"] = emoji_images["smile"]

    # If plain is missing, fall back to smile/sad as a neutral-ish default
    if emoji_images["plain"] is None:
        if emoji_images["smile"] is not None:
            emoji_images["plain"] = emoji_images["smile"]
        elif emoji_images["sad"] is not None:
            emoji_images["plain"] = emoji_images["sad"]

    print("\n" + "="*60)
    print("📁 Loaded emoji images:")
    print("="*60)
    image_mapping = {
        "smile": "smile.jpg",
        "surprised": "surprised.jpg",
        "sad": "sad.jpg",
        "dance": "dance.jpg",
        "air": "air.jpg",
        "heart": "heart.jpg",
        "cat": "cat.jpg",
        "plain": "plain.png (or smile/sad fallback)",
        "crazy": "crazy.jpg / wave.jpg / wink.jpg / surprised.jpg / smile.jpg",
        "thinking": "thinking.jpg",
        "middle": "middle.jpg",
        "wave": "wave.jpg",
        "wink": "wink.jpg",
    }
    for key, img in emoji_images.items():
        status = "✓" if img is not None else "✗"
        file_info = image_mapping.get(key, f"{key}.jpg")
        print(f"  {status} {key:12s} → {file_info}")
    print("="*60)

except Exception as e:
    print(f"Error loading emoji images: {e}")
    print("\nExpected files (at least some of them):")
    print("- smile.jpg, surprised.jpg, sad.jpg, plain.png")
    print("- dance.jpg, air.jpg, heart.jpg, crazy.jpg, thinking.jpg, middle.jpg")
    print("- wave.jpg, wink.jpg (optional fallbacks for crazy)")
    exit()

# Create a blank fallback emoji with correct (height, width) ordering
blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[1], EMOJI_WINDOW_SIZE[0], 3), dtype=np.uint8)

def resize_camera_frame(frame: np.ndarray) -> np.ndarray:
    """Resize camera frame to the configured window size."""
    return cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

def resize_emoji(img: np.ndarray | None) -> np.ndarray:
    """Resize emoji image to the configured window size, falling back to blank."""
    if img is None:
        return blank_emoji.copy()
    return cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)**0.5

# MediaPipe Hands landmark indices
HAND_WRIST = 0
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20
HAND_MIDDLE_MCP = 9

def detect_heart_gesture(hand_landmarks):
    """Detect if hands form a heart shape (thumbs + index fingers close together)."""
    if len(hand_landmarks) < 2:
        return False
    
    # Get thumb tips and index finger tips from both hands
    hand1_thumb = hand_landmarks[0].landmark[HAND_THUMB_TIP]
    hand1_index = hand_landmarks[0].landmark[HAND_INDEX_TIP]
    hand2_thumb = hand_landmarks[1].landmark[HAND_THUMB_TIP]
    hand2_index = hand_landmarks[1].landmark[HAND_INDEX_TIP]
    
    # Check distance between thumbs and between index fingers
    thumb_distance = calculate_distance(hand1_thumb, hand2_thumb)
    index_distance = calculate_distance(hand1_index, hand2_index)
    
    # Also check if thumbs and index fingers of same hand are close (curved shape)
    hand1_thumb_index_dist = calculate_distance(hand1_thumb, hand1_index)
    hand2_thumb_index_dist = calculate_distance(hand2_thumb, hand2_index)
    
    # Check if hands are near face area (y coordinate should be above center of image)
    avg_y = (hand1_thumb.y + hand2_thumb.y) / 2
    
    # Heart gesture: thumbs close together, index fingers close together, 
    # and hands are curved (thumb-index distance is reasonable for curved shape)
    thumbs_close = thumb_distance < HEART_DISTANCE_THRESHOLD
    indices_close = index_distance < HEART_DISTANCE_THRESHOLD
    hands_curved = hand1_thumb_index_dist < 0.12 and hand2_thumb_index_dist < 0.12
    near_face = avg_y < 0.6  # More lenient threshold for face area
    
    return thumbs_close and indices_close and hands_curved and near_face

def detect_finger_heart(hand_landmarks):
    """Detect a one-handed "finger heart" gesture (Korean heart).

    Rough heuristic: thumb tip and index tip very close together, other fingers curled.
    """
    for hand in hand_landmarks:
        thumb_tip = hand.landmark[HAND_THUMB_TIP]
        index_tip = hand.landmark[HAND_INDEX_TIP]

        # MCP joints for fingers
        index_mcp = hand.landmark[5]
        middle_tip = hand.landmark[HAND_MIDDLE_TIP]
        middle_mcp = hand.landmark[9]
        ring_tip = hand.landmark[HAND_RING_TIP]
        ring_mcp = hand.landmark[13]
        pinky_tip = hand.landmark[HAND_PINKY_TIP]
        pinky_mcp = hand.landmark[17]

        # Thumb and index tips should be very close together
        thumb_index_dist = calculate_distance(thumb_tip, index_tip)
        thumb_index_close = thumb_index_dist < 0.03

        # Other fingers should be curled (tips not far above MCPs)
        middle_curled = middle_tip.y > middle_mcp.y - 0.01
        ring_curled = ring_tip.y > ring_mcp.y - 0.01
        pinky_curled = pinky_tip.y > pinky_mcp.y - 0.01

        if thumb_index_close and middle_curled and ring_curled and pinky_curled:
            return True
    return False

def detect_middle_finger(hand_landmarks):
    """Detect if middle finger is raised."""
    for hand_landmark in hand_landmarks:
        # Get key points using indices
        wrist = hand_landmark.landmark[HAND_WRIST]
        middle_tip = hand_landmark.landmark[HAND_MIDDLE_TIP]
        middle_mcp = hand_landmark.landmark[HAND_MIDDLE_MCP]
        
        # Get other finger tips
        index_tip = hand_landmark.landmark[HAND_INDEX_TIP]
        ring_tip = hand_landmark.landmark[HAND_RING_TIP]
        pinky_tip = hand_landmark.landmark[HAND_PINKY_TIP]
        thumb_tip = hand_landmark.landmark[HAND_THUMB_TIP]
        
        # Middle finger should be highest (lowest y value)
        # Other fingers should be lower
        finger_tips = [index_tip, ring_tip, pinky_tip, thumb_tip]
        middle_is_highest = all(middle_tip.y < tip.y for tip in finger_tips)
        
        # Middle finger should be extended (tip significantly above MCP)
        middle_extended = middle_tip.y < middle_mcp.y - 0.05
        
        if middle_is_highest and middle_extended:
            return True
    return False

def is_fist_closed(hand_landmark):
    """Detect if hand is in a closed fist (fingers curled)."""
    # Get finger tips and their corresponding MCP (metacarpophalangeal) joints
    index_tip = hand_landmark.landmark[HAND_INDEX_TIP]
    index_mcp = hand_landmark.landmark[5]  # INDEX_FINGER_MCP
    middle_tip = hand_landmark.landmark[HAND_MIDDLE_TIP]
    middle_mcp = hand_landmark.landmark[9]  # MIDDLE_FINGER_MCP
    ring_tip = hand_landmark.landmark[HAND_RING_TIP]
    ring_mcp = hand_landmark.landmark[13]  # RING_FINGER_MCP
    pinky_tip = hand_landmark.landmark[HAND_PINKY_TIP]
    pinky_mcp = hand_landmark.landmark[17]  # PINKY_MCP
    
    # In a closed fist, finger tips are closer to (or below) their MCP joints
    # We check if tips are below (higher y value) or very close to MCP joints
    index_curled = index_tip.y > index_mcp.y - 0.02
    middle_curled = middle_tip.y > middle_mcp.y - 0.02
    ring_curled = ring_tip.y > ring_mcp.y - 0.02
    pinky_curled = pinky_tip.y > pinky_mcp.y - 0.02
    
    # All fingers should be curled for a closed fist
    return index_curled and middle_curled and ring_curled and pinky_curled

def main() -> None:
    """Main application entrypoint for the Emoji Reactor."""
    global current_emotion, last_emotion

    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow('Camera Feed', 100, 100)
    cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

    print("\n" + "="*60)
    print("🎭 EMOJI REACTOR - Ready!")
    print("="*60)
    print("\nReactions to try:")
    print("  1. 😊 Smile/Laugh - Show teeth, raise cheeks")
    print("  2. 😲 Surprised - Mouth wide open, eyes wide")
    print("  3. 😢 Sad - Mouth corners down, eyes half-closed")
    print("  4. 💃 Dance - Both hands up with CLOSED FISTS")
    print("  5. 🙌 Air - Both hands up with OPEN FISTS")
    print("  6. ❤️/🐱 Heart / Cat - Two-hand heart or Korean finger-heart")
    print("  7. 🤪 Crazy/Silly - Tilt head + one eye smaller + mouth open")
    print("  8. 🤔 Thinking - Tilt head + hand near lips")
    print("  9. 🖕 Middle Finger - Single raised middle finger")
    print("\nPress 'q' to quit")
    print("="*60)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            detected_state = "PLAIN"  # Default state
            
            # Priority 1: Check hand-based gestures first
            results_hands = hands.process(image_rgb)
            results_pose = pose.process(image_rgb)
            
            hand_gesture_detected = False
            
            # Check for heart gestures (two-hand or one-hand finger heart)
            if results_hands.multi_hand_landmarks:
                if (len(results_hands.multi_hand_landmarks) >= 2 and
                        detect_heart_gesture(results_hands.multi_hand_landmarks)) or \
                   detect_finger_heart(results_hands.multi_hand_landmarks):
                    detected_state = "CAT"
                    hand_gesture_detected = True
                
                # Check for middle finger
                if not hand_gesture_detected and results_hands.multi_hand_landmarks:
                    if detect_middle_finger(results_hands.multi_hand_landmarks):
                        detected_state = "MIDDLE"
                        hand_gesture_detected = True
                
                # Check for dance (both hands up with closed fists) vs air (open fists)
                if not hand_gesture_detected and results_pose.pose_landmarks and results_hands.multi_hand_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    left_hand_up = left_wrist.y < left_shoulder.y - HAND_ABOVE_SHOULDER_OFFSET
                    right_hand_up = right_wrist.y < right_shoulder.y - HAND_ABOVE_SHOULDER_OFFSET
                    
                    if left_hand_up and right_hand_up:
                        # Check if both hands are closed fists
                        left_fist = False
                        right_fist = False
                        
                        # Determine which hand is which based on position
                        for hand_landmark in results_hands.multi_hand_landmarks:
                            wrist = hand_landmark.landmark[HAND_WRIST]
                            
                            # Check if this is left or right hand by comparing with pose landmarks
                            if abs(wrist.x - left_wrist.x) < abs(wrist.x - right_wrist.x):
                                left_fist = is_fist_closed(hand_landmark)
                            else:
                                right_fist = is_fist_closed(hand_landmark)
                        
                        # If both hands are closed fists -> DANCE, if open -> AIR
                        if left_fist and right_fist:
                            detected_state = "DANCE"
                        else:
                            detected_state = "AIR"
                        hand_gesture_detected = True
            
            # Process face mesh once (needed for both thinking detection and facial expressions)
            results_face = face_mesh.process(image_rgb)
            
            # Check for thinking (head tilted + hand near lips)
            if not hand_gesture_detected and results_face.multi_face_landmarks and results_hands.multi_hand_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Get head tilt angle
                    nose_tip = face_landmarks.landmark[1]
                    chin_point = face_landmarks.landmark[175]
                    forehead = face_landmarks.landmark[10]
                    
                    face_vertical = abs(forehead.y - chin_point.y)
                    face_horizontal = abs(forehead.x - chin_point.x)
                    head_tilt_angle = np.degrees(np.arctan(face_horizontal / face_vertical)) if face_vertical > 0 else 0
                    
                    # Get mouth/lip landmarks
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    mouth_center = face_landmarks.landmark[14]  # Lower lip center
                    
                    # Check if head is tilted
                    if head_tilt_angle > CRAZY_HEAD_TILT_THRESHOLD:
                        # Check if hand is near lips
                        for hand_landmark in results_hands.multi_hand_landmarks:
                            wrist = hand_landmark.landmark[HAND_WRIST]
                            index_tip = hand_landmark.landmark[HAND_INDEX_TIP]
                            
                            # Check distance to mouth/lips
                            wrist_to_mouth = calculate_distance(wrist, mouth_center)
                            index_to_mouth = calculate_distance(index_tip, mouth_center)
                            
                            if wrist_to_mouth < CHIN_HAND_DISTANCE_THRESHOLD or index_to_mouth < CHIN_HAND_DISTANCE_THRESHOLD:
                                detected_state = "THINKING"
                                hand_gesture_detected = True
                                break
                    if hand_gesture_detected:
                        break
            
            # Priority 2: Check facial expressions if no hand gesture detected
            if not hand_gesture_detected:
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        # Mouth landmarks
                        left_corner = face_landmarks.landmark[61]
                        right_corner = face_landmarks.landmark[291]
                        upper_lip = face_landmarks.landmark[13]
                        lower_lip = face_landmarks.landmark[14]
                        
                        mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                        mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
                        
                        # Mouth corners position
                        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
                        corner_avg_y = (left_corner.y + right_corner.y) / 2
                        mouth_corner_down = corner_avg_y - mouth_center_y
                        
                        # Eye landmarks
                        left_eye_top = face_landmarks.landmark[159]
                        left_eye_bottom = face_landmarks.landmark[145]
                        left_eye_left = face_landmarks.landmark[33]
                        left_eye_right = face_landmarks.landmark[133]
                        
                        right_eye_top = face_landmarks.landmark[386]
                        right_eye_bottom = face_landmarks.landmark[374]
                        right_eye_left = face_landmarks.landmark[362]
                        right_eye_right = face_landmarks.landmark[263]
                        
                        # Calculate eye aspect ratios
                        left_eye_height = abs(left_eye_bottom.y - left_eye_top.y)
                        left_eye_width = abs(left_eye_right.x - left_eye_left.x)
                        right_eye_height = abs(right_eye_bottom.y - right_eye_top.y)
                        right_eye_width = abs(right_eye_right.x - right_eye_left.x)
                        
                        left_eye_ratio = left_eye_height / left_eye_width if left_eye_width > 0 else 1.0
                        right_eye_ratio = right_eye_height / right_eye_width if right_eye_width > 0 else 1.0
                        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
                        eye_ratio_diff = abs(left_eye_ratio - right_eye_ratio)
                        
                        # Head tilt detection
                        nose_tip = face_landmarks.landmark[1]
                        chin_point = face_landmarks.landmark[175]
                        forehead = face_landmarks.landmark[10]
                        
                        face_vertical = abs(forehead.y - chin_point.y)
                        face_horizontal = abs(forehead.x - chin_point.x)
                        head_tilt_angle = np.degrees(np.arctan(face_horizontal / face_vertical)) if face_vertical > 0 else 0
                        
                        # Detection logic for each expression
                        
                        # Crazy/Silly: Head tilted + one eye smaller + mouth wide open
                        if (head_tilt_angle > CRAZY_HEAD_TILT_THRESHOLD and 
                            eye_ratio_diff > CRAZY_EYE_DIFF_THRESHOLD and 
                            mouth_aspect_ratio > SURPRISED_MOUTH_THRESHOLD):
                            detected_state = "CRAZY"
                        
                        # Surprised: Mouth wide open, eyes open wide
                        elif (mouth_aspect_ratio > SURPRISED_MOUTH_THRESHOLD and 
                              avg_eye_ratio > SURPRISED_EYE_THRESHOLD):
                            detected_state = "SURPRISED"
                        
                        # Sad: Mouth corners down, eyes half-closed
                        elif (mouth_corner_down > SAD_MOUTH_THRESHOLD and 
                              avg_eye_ratio < SAD_EYE_THRESHOLD):
                            detected_state = "SAD"
                        
                        # Smile/Laugh: High mouth aspect ratio
                        elif mouth_aspect_ratio > SMILE_THRESHOLD:
                            detected_state = "SMILE"
                        
                        else:
                            detected_state = "PLAIN"
            
            # State machine to avoid flickering using a small history buffer
            state_buffer.append(detected_state)
            if len(state_buffer) > 0:
                # Choose the most common state in the recent window
                most_common_state, _ = Counter(state_buffer).most_common(1)[0]
                current_emotion = most_common_state
            else:
                current_emotion = detected_state
            last_emotion = detected_state

            # Select emoji based on state
            emoji_map = {
                "SMILE": ("smile", "😊"),
                "SURPRISED": ("surprised", "😲"),
                "SAD": ("sad", "😢"),
                "DANCE": ("dance", "💃"),
                "AIR": ("air", "🙌"),
                "HEART": ("heart", "❤️"),
                "CAT": ("cat", "🐱"),
                "CRAZY": ("crazy", "🤪"),
                "THINKING": ("thinking", "🤔"),
                "MIDDLE": ("middle", "🖕"),
                "PLAIN": ("plain", "😐"),
            }
            
            emoji_key, emoji_name = emoji_map.get(current_emotion, ("plain", "😐"))
            emoji_to_display = emoji_images.get(emoji_key)

            if emoji_to_display is None:
                # Fallback to plain if emoji not found
                emoji_to_display = emoji_images.get("plain", blank_emoji)

            # Draw pose landmarks (skeleton) on the frame if available
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2),
                )

            # Draw hand landmarks if available
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2, circle_radius=2),
                    )

            # Draw face mesh landmarks if available
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                    )

            camera_frame_resized = resize_camera_frame(frame)
            emoji_to_display_resized = resize_emoji(emoji_to_display)

            # Overlay a small emoji thumbnail in the top-right corner of the camera feed
            thumb_size = 120
            emoji_thumb = cv2.resize(emoji_to_display_resized, (thumb_size, thumb_size))
            y1, y2 = 10, 10 + thumb_size
            x2, x1 = WINDOW_WIDTH - 10, WINDOW_WIDTH - 10 - thumb_size
            roi = camera_frame_resized[y1:y2, x1:x2]
            if roi.shape[:2] == emoji_thumb.shape[:2]:
                camera_frame_resized[y1:y2, x1:x2] = emoji_thumb

            # Display state with emoji
            state_text = f'STATE: {current_emotion} {emoji_name}'
            cv2.putText(camera_frame_resized, state_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(camera_frame_resized, "Press 'q' to quit", (10, WINDOW_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera Feed', camera_frame_resized)
            cv2.imshow('Emoji Output', emoji_to_display_resized)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
