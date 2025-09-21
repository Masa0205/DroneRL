import threading
import queue
import time
import numpy as np
import airsim


def image_acquisition_thread(image_queue):
    airsim_client = airsim.MultirotorClient()
    airsim_client.confirmConnection()
    while True:
        try:
            # AirSimから最新の画像を非同期で取得
            responses = airsim_client.simGetImages([airsim.client.ImageRequest("0", airsim.client.ImageType.Scene, False, False)])
            image_response = responses[0]
            
            # 画像データをnumpy配列に変換
            img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)
            
            # キューが満杯なら一番古い画像を破棄
            if image_queue.full():
                image_queue.get()
                
            # 最新の画像をキューに追加
            image_queue.put(img_rgb)
            #print("success")
        except Exception as e:
            print("getImageError", e)
            # 取得間隔を調整（例: 0.03秒 = 約33fps）
        time.sleep(0.03)
