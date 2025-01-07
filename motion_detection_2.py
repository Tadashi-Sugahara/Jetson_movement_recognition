import cv2
import numpy as np
import time
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_gesture(landmarks):
    nose = landmarks[0]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # 両手を頭の上で組んで丸を作っている動作の検出
    if left_wrist.y < nose.y and right_wrist.y < nose.y:
        if np.linalg.norm(np.array([left_wrist.x, left_wrist.y]) - np.array([right_wrist.x, right_wrist.y])) < 0.3:
            return 'Correct'

    # 手が映っていて、左手が右肩で右手が左肩に接近しているポーズの検出
    if left_wrist.visibility > 0.3 and right_wrist.visibility > 0.3:
        if left_wrist.x > right_shoulder.x and right_wrist.x < left_shoulder.x:
            return 'Incorrect'

    # 両手が映っていない場合
    if left_wrist.visibility < 0.3 and right_wrist.visibility < 0.3:
        return 'No Gesture'

    return 'No Gesture'

def run_motion_detection():
    cap = cv2.VideoCapture(1)
    gesture_detected = 'No Gesture'
    gesture_count = 0
    last_gesture = None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        time.sleep(3)

        ret, frame = cap.read()
        if not ret:
            break

        # 顔検出
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 1:
            print("NG: Multiple people")
            continue
        elif len(faces) == 1:
            print("OK: Single person detected")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            gesture_detected = detect_gesture(results.pose_landmarks.landmark)
            cv2.putText(frame, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if gesture_detected == 'Correct' else (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Motion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_motion_detection()