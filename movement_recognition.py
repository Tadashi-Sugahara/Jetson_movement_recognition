import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# MoveNetモデルのダウンロードと読み込み
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

# Webカメラの映像を取得
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

def detect_pose(keypoints):
    # キーポイントの取得
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # ガッツポーズの検知
    if (left_wrist[2] > 0.3 and right_wrist[2] > 0.3 and
        left_elbow[2] > 0.3 and right_elbow[2] > 0.3 and
        left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3):
        
        if (left_wrist[1] < left_shoulder[1] + 20 and right_wrist[1] < right_shoulder[1] + 20 and
            left_elbow[1] < left_shoulder[1] + 20 and right_elbow[1] < right_shoulder[1] + 20):
            return "Guts Pose"

    # ピースサインの検知（簡単な例として、右手が顔の近くにある場合）
    if (right_wrist[2] > 0.3 and right_elbow[2] > 0.3 and right_shoulder[2] > 0.3):
        if (right_wrist[1] < right_shoulder[1] and right_elbow[1] < right_shoulder[1]):
            return "Peace Sign"

    return "Unknown"

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # フレームをリサイズして処理速度を向上させる
    input_image = cv2.resize(frame, (192, 192))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.convert_to_tensor(input_image, dtype=tf.int32)

    # 動作認識モデルを使用してポーズ推定
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # キーポイントを描画
    for keypoint in keypoints:
        y, x, confidence = keypoint
        if confidence > 0.3:  # 信頼度の閾値を下げる
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # ポーズを検知して表示
    pose = detect_pose(keypoints)
    cv2.putText(frame, pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()