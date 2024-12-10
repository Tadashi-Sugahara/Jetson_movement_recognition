import tensorflow as tf
import numpy as np
import cv2

# MoveNetモデルのダウンロードと読み込み
model = tf.saved_model.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')

# Webカメラの映像を取得
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

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
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # 動作認識モデルを使用してポーズ推定
    outputs = model(input_image)
    keypoints = outputs['output_0'].numpy()

    # キーポイントを描画
    for keypoint in keypoints[0, 0, :, :]:
        x, y, confidence = keypoint
        if confidence > 0.5:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

    # 結果を表示
    cv2.imshow('Pose Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()