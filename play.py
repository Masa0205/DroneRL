import pickle
import random
import time
import threading
from collections import deque
import queue
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt

from img import arUco, wrapper
from agent import replay_buffer, Agent
from control import Control
from sub import image_acquisition_thread

def reward(images, time):
    img = images[-1]
    centers = aruco.detect(img)
    alpha = 1
    beta = 0.5
    if centers is not None and len(centers) > 0:
        H, W = img.shape[:2]
        cx_img, cy_img = W // 2, H // 2  # 画像中心
        # 画像中心
        cx_img, cy_img = W//2, H//2
        # マーカー座標
        cx, cy = centers[0]
        # 誤差
        dx, dy = cx - cx_img, cy - cy_img
        dist = np.sqrt(dx**2 + dy**2)

        # 最大距離
        d_max = np.sqrt((W/2)**2 + (H/2)**2)

        # 正規化
        norm_dist = dist / d_max

        # 報酬中心に近いほど高い
        r_pos = 1 - norm_dist 
        # 時間が経過するほど高い
        r_time = time / 30
        r = alpha * r_pos + beta * r_time
        done = False
    else:
        r = -1
        done = True
    if time >= 30:
        r += 1
        done = True
    #print(f"r = {r}, done = {done}")
    return r, done


def clip(image_queue, local_images):
    while not image_queue.empty():
            try:
                img = image_queue.get_nowait()
                local_images.append(img)
            except queue.Empty:
                # 例外処理: 複数のスレッドが同時にアクセスした場合の安全対策
                break
    return local_images

def main():
    t = 0
    episodes = 10
    memory = replay_buffer(N, n)
    update_interval = 50
    print_interval = 10
    save_interval = 1000
    loss_list = []
    score_lst = []
    time_list = []
    total_score = 0.0
    total_time = 0.0
    loss = 0.0
    eps = 1.0
    eps_end = 0.05
    eps_decay = 0.995  # 1 エピソードごとの減衰率
    p = 0
    # 画像取得キュー
    image_queue = queue.Queue(maxsize=4) # 4フレームを保持
    local_images = []
    client = C.client
    for k in range(1, episodes+1):
        #print(len(memory))
        eps = 0
        #print(f"eps = {eps}")
        if k == 1:
            # 画像取得スレッドを起動
            image_thread = threading.Thread(target=image_acquisition_thread, args=(image_queue,))
            image_thread.daemon = True # メインスレッド終了時にスレッドも終了
            image_thread.start()
        # 2. ★★★ キューを初期化する ★★★
        # 前のエピソードの残りフレームをすべて破棄し、キューを空にする
        while not image_queue.empty():
            try:
                # 空でも待機しない get_nowait() を使うのが安全
                #中身がある限り取り出し続けて空にする動作
                image_queue.get_nowait()
            except queue.Empty:
                # 例外処理: 複数のスレッドが同時にアクセスした場合の安全対策
                break
            
        C.reset() #環境初期化

        #この瞬間の4フレームを切り取る
        local_images = clip(image_queue, local_images)
        s = W.preprocess(local_images)
        done = False
        start_time = time.perf_counter()
        while not done:

            a = agent.action(s, eps)
            #print(f"a = {a}")
            if a == 0:
                C.hovering()
                print("hover")
            elif a == 1:
                C.left()
                print("left")
            elif a == 2:
                C.right()
                print("right")
            current_time = time.perf_counter() - start_time
            local_images = clip(image_queue, local_images)
            r, done = reward(local_images, current_time)
            s_prime = W.preprocess(local_images)
            #print(f"r={r}")
            total_score += r
            s = s_prime
            if done:
                total_time += current_time
            

N = 50000
n = 256

#インポートでなく直接実行したときのみ処理を走らせるためのもの
if __name__ == "__main__":
    W = wrapper()
    aruco = arUco()
    C = Control()
    agent = Agent()
    name = input("使用するモデル名(パス)：")
    agent.load(name)

    main()