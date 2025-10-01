import cv2
import numpy as np
import torch
from collections import deque
import copy
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
        area = None
        centers = []
        output_img = image.copy()

        if ids is not None:
            for i, corner in enumerate(corners):
                # corner: (1, 4, 2) → 4つの頂点座標
                pts = corner[0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                centers = [cx, cy]
                #面積
                area = cv2.contourArea(pts)
                if draw:
                    cv2.aruco.drawDetectedMarkers(output_img, corners, ids)
                    cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
        #print(centers)
        return centers, area, output_img
    
    def get_state(self, obs, a=np.array([0.0, 0.0], dtype=np.float32)):
        agent_pos = np.array(obs,dtype=np.float32)
        target_pos = np.array([480, 360],dtype=np.float32)

        s_x, s_y = target_pos - agent_pos
        s_x, s_y = s_x/960, s_y/720
        state = np.array([s_x, s_y], dtype=np.float32)
        state = np.hstack((state,a))
        #print(state)
        return state
