import cv2
import mediapipe as mp
import math
import numpy as np
import sys


###############
#BOX FUNCTIONS#
###############

def gaussBlur(frame, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Ensure valid bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]

    # Apply blur
    blurred_roi = cv2.GaussianBlur(roi, (101, 101), 4)

    # Replace ROI in original frame
    frame[y1:y2, x1:x2] = blurred_roi

def cameraBlur(frame, top_left, bottom_right):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)

    alpha = mask[:, :, None] / 255.0
    frame[:] = (blurred * alpha + frame * (1 - alpha)).astype(np.uint8)

def cameraLens(frame, top_left, bottom_right, zoom_factor=2):
    x1, y1 = top_left
    x2, y2 = bottom_right

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]

    zoomed = cv2.resize(
        roi,
        None,
        fx=zoom_factor,
        fy=zoom_factor,
        interpolation=cv2.INTER_LINEAR
    )

    # Crop center
    zh, zw = zoomed.shape[:2]
    start_y = (zh - h) // 2
    start_x = (zw - w) // 2

    cropped = zoomed[start_y:start_y + h, start_x:start_x + w]

    # Safety resize (prevents broadcasting error)
    cropped = cv2.resize(cropped, (w, h))

    frame[y1:y2, x1:x2] = cropped

def swirlEffect(frame, top_left, bottom_right, strength=2.0):
    x1, y1 = top_left
    x2, y2 = bottom_right

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]
    cx, cy = w // 2, h // 2

    y, x = np.indices((h, w))
    x -= cx
    y -= cy

    r = np.sqrt(x**2 + y**2)
    angle = strength * r / max(w, h)

    new_x = x * np.cos(angle) - y * np.sin(angle) + cx
    new_y = x * np.sin(angle) + y * np.cos(angle) + cy

    swirled = cv2.remap(
        roi,
        new_x.astype(np.float32),
        new_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    frame[y1:y2, x1:x2] = swirled

EFFECTS = {
    "gauss": gaussBlur,
    "camera": cameraBlur,
    "lens": cameraLens,
    "swirl": swirlEffect
}

effect_name = sys.argv[1] if len(sys.argv) > 1 else "gauss"
effect_func = EFFECTS.get(effect_name, gaussBlur)

######################
#END OF BOX FUNCTIONS#
######################

# ---------------------------
# MediaPipe setup
# ---------------------------
mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------------------
# Constants
# ---------------------------
PINCH_THRESHOLD = 0.4

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    h, w, _ = frame.shape
    pinch_points = []  # stores (x_px, y_px)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ---- Required landmarks ----
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # ---- Hand size ----
            hand_size = math.hypot(
                index_mcp.x - wrist.x,
                index_mcp.y - wrist.y
            )

            # ---- Pinch distance ----
            pinch_distance = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            relative_pinch = pinch_distance / hand_size
            pinch_detected = relative_pinch < PINCH_THRESHOLD

            if pinch_detected:
                # Pinch midpoint (normalized)
                pinch_x = (thumb_tip.x + index_tip.x) / 2
                pinch_y = (thumb_tip.y + index_tip.y) / 2

                # Convert to pixels
                px = int(pinch_x * w)
                py = int(pinch_y * h)

                pinch_points.append((px, py))

                # Draw pinch point
                cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)

    # ---------------------------
    # Draw box if TWO pinches exist
    # ---------------------------
    if len(pinch_points) == 2:
        (x1, y1), (x2, y2) = pinch_points

        top_left = (min(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))

        #distortion here
        effect_func(frame, top_left, bottom_right)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 5)

        cv2.putText(
            frame,
            "TWO-HAND PINCH BOX",
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow("Two-Hand Pinch Box", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

