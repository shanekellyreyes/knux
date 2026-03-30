import cv2
import mediapipe as mp
import math
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MATH BRAIN ---
def calculate_angle(a, b, c):
    """Calculates the interior angle at joint B."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# --- SETUP ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture('test_video.mp4')
paused = False
chin_buffer = deque(maxlen=10) # Smooths out the "Chin Up/Down" flickering

# Full Skeleton Mapping
SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso
    (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
]

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            # Map all landmarks to pixel coordinates
            pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))}

            # 1. DRAW SKELETON (Bones)
            for start, end in SKELETON:
                cv2.line(frame, pts[start], pts[end], (255, 255, 255), 2)

            # 2. DRAW JOINTS (The Green Dots)
            for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]:
                cv2.circle(frame, pts[i], 5, (0, 255, 0), -1)

            # 3. CALCULATE & DISPLAY LIMB ANGLES
            # Format: { Label: (Point A, Joint B, Point C) }
            joints = {
                "R_Elb": (pts[12], pts[14], pts[16]),
                "L_Elb": (pts[11], pts[13], pts[15]),
                "R_Kne": (pts[24], pts[26], pts[28]),
                "L_Kne": (pts[23], pts[25], pts[27])
            }
            
            for name, (a, b, c) in joints.items():
                angle = calculate_angle(a, b, c)
                # Place the number right next to the joint (B)
                cv2.putText(frame, str(int(angle)), (b[0] + 10, b[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 4. STEADY CHIN LOGIC
            mid_shoulder = ( (pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2 )
            mid_hip = ( (pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2 )
            spine_len = math.dist(mid_shoulder, mid_hip)
            nose_offset = mid_shoulder[1] - pts[0][1]
            
            # Threshold: (Lower = stricter, Higher = more relaxed)
            chin_buffer.append( (nose_offset / spine_len) > 0.25 )
            is_chin_up = sum(chin_buffer) > (len(chin_buffer) * 0.7)
            
            color = (0, 0, 255) if is_chin_up else (0, 255, 0)
            status = "CHIN UP!" if is_chin_up else "CHIN TUCKED"
            
            cv2.rectangle(frame, (10, 10), (320, 60), (0,0,0), -1)
            cv2.putText(frame, f"{status}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 5. UI & CONTROLS
    cv2.imshow("Kinematics Engine v3.2", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' '): 
        paused = not paused
        if paused:
            cv2.putText(frame, "PAUSED", (int(w/2)-80, int(h/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow("Kinematics Engine v3.2", frame)

detector.close()
cap.release()
cv2.destroyAllWindows()