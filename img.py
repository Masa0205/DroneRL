import cv2
import numpy as np
import torch
from collections import deque
import queue
class arUco:
    def __init__(self, dictionary=cv2.aruco.DICT_4X4_50):
        """
        ArUcoマーカー検出クラス
        :param dictionary: 使用するマーカー辞書 (例: cv2.aruco.DICT_4X4_50)
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()

    def detect(self, image, draw=True):
        """
        画像からArUcoを検出し、中心座標を返す
        :param image: 入力画像 (BGR形式)
        :param draw: マーカーを描画するかどうか
        :return: (output_img, centers)
                 output_img: 検出済み画像 (BGR)
                 centers: [ (cx, cy), ...]
        """
        if image is None or image.size == 0:
            print("⚠️ detect() に空の画像が渡されました。")
            return []
        # グレースケールに変換
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("cv2.cvtColor エラー:", e)
            return []
        
        

        # マーカー検出
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        centers = []
        output_img = image.copy()

        if ids is not None:
            for i, corner in enumerate(corners):
                # corner: (1, 4, 2) → 4つの頂点座標
                pts = corner[0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                centers.append((cx, cy))

                if draw:
                    cv2.aruco.drawDetectedMarkers(output_img, corners, ids)
                    cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
        #print(centers)
        return centers
class wrapper:
    def __init__(self, height=84, width=84, gray=True, device="cpu", n_stack = 4):
        self.height = height
        self.width = width
        self.gray = gray
        self.device = device
        self.n_stack = n_stack
        self.frames = deque([], maxlen=n_stack)
    def preprocess(self, local_images):
        """
        input local_images[]: H x W x C (RGB形式, numpy配列) * 4
        return: [1, C, H, W] の torch.Tensor
        """
        #0枚の場合黒画像を入れる
        if not local_images:
            img = np.zeros((720, 960, 3), dtype=np.uint8)
            for _ in range(4):
                local_images.append(img)  # 黒画像
        #4枚以上なら不足分捨てる、足りてなかったら最初のフレームを複製しパディング
        if len(local_images) >= 4:
            state_images = local_images[-4:]
        else:
            padding_image = local_images[0]
            padding_num = 4 - len(local_images)
            state_images = [padding_image] * padding_num + local_images
        for img in state_images:
            # 1. グレースケール化
            if len(img.shape) == 3:  # (高さ, 幅, チャンネル) → 3チャンネル以上
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 2. リサイズ
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # 3. 正規化 (0-1)
            img = img.astype(np.float32) / 255.0

            # 4. チャンネル順変更
            if self.gray:
                img = np.expand_dims(img, axis=0)  # (H, W) → (1, H, W)
            else:
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)

            # フレーム保存
            self.frames.append(img)
        

        # (n_stack*C, H, W)
        stacked = np.concatenate(list(self.frames), axis=0)

        # (1, n_stack*C, H, W)
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(self.device)

        return tensor

        