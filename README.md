# Jetson_movement_necognition
Jetsonを使って動体検知(ガッツポーズやピースサイン）


# 使用パッケージ
numpy,
opencv-python,
tensorflow,
tensorflow_hub,

# Webカメラの設定 ()内の数値でWebカメラの選択　
USBカメラの場合は、(1)
# Webカメラの映像を取得
cap = cv2.VideoCapture(1)
