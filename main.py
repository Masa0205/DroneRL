import pickle
import random
import time
import threading
from collections import deque
import queue
import os
import pickle
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
    alpha = 1
    beta = 0.5
    for i, img in enumerate(reversed(images)):
        centers = aruco.detect(img)
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
            #print(f"{i}番目発見")
            break
        else:
            if i == len(images)-1:
                r = -1
                done = True
                #print("発見できず")
    if time >= 10:
        r += 1
        done = True
    #print(f"r = {r}, done = {done}")
    return r, done

def graph_score(score_lst, time_list):
    now = datetime.now()
    date = now.date()
    x_axis = np.arange(len(score_lst))

    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    # 実験名＋タイムスタンプでファイル名を自動生成
    experiment_name = "Drone_DQN"
    data_type = "av per 10 eps"
    filename = f"{experiment_name}_reward_{date}.png"
    save_path = os.path.join(save_dir, filename)

    # グラフ作成
    plt.plot(x_axis, score_lst)
    plt.xlabel("av per 10 eps")
    plt.ylabel("reward")

    # 保存
    plt.savefig(save_path)
    plt.close()  # メモリ節約のために閉じる

 
    print(f"実験結果を保存しました:")

def clip(image_queue, clip_images):
    while not image_queue.empty():
            try:
                img = image_queue.get_nowait()
                if img is None or not hasattr(img, "shape") or img.size == 0:
                    continue
                clip_images.append(img)
            except queue.Empty:
                # 例外処理: 複数のスレッドが同時にアクセスした場合の安全対策
                break
    return clip_images

def main():
    t = 0
    episodes = 10000
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
    eps_start = 1
    eps_end = 0.05
    eps_decay = (eps_end / eps_start) ** (1 / episodes)
    eps = eps_start
    p = 0
    # 画像取得キュー
    image_queue = queue.Queue(maxsize=4) # 過去4フレームを保持
    #img_buffer = deque(maxlen=4)
    local_images = []
    #client = C.client
    data_path = "my_replay_buffer.pkl"
    #オフライン学習
    if not os.path.exists(data_path):
        print("バッファーデータがないです")
    else:
        try:
            with open(data_path, "rb") as f:
                memory = pickle.load(f)

            print("読み込み成功")
        except Exception as e:
            print(f"読み込みエラー{e}")
    for k in range(1, episodes+1):
        #print(len(memory))
        eps = max(eps_end, eps * eps_decay)
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
        clip_images = clip(image_queue, [])
        s = W.preprocess(local_images)
        done = False
        start_time = time.perf_counter()
        while not done:

            a = agent.action(s, eps)
            #print(f"a = {a}")
            if a == 0:
                C.hovering()
                #print("hover")
            elif a == 1:
                C.left()
                #print("left")
            elif a == 2:
                C.right()
                #print("right")
            current_time = time.perf_counter() - start_time
            local_images = clip(image_queue, [])
            #for img in local_images:
                #img_buffer.append(img)
            #print(len(local_images))
            r, done = reward(local_images, current_time)
            s_prime = W.preprocess(local_images)
            #print(f"r={r}")
            total_score += r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            if done:
                total_time += current_time
            
            if len(memory) > 2000:
                if p == 0:
                    # プログラム終了前にファイルへ保存
                    with open('my_replay_buffer.pkl', 'wb') as f:
                        pickle.dump(memory, f)
                    print("学習スタート")
                    p += 1
                
                loss += agent.train(memory)
                t += 1
            if t % update_interval == 0:
                agent.copy_weights()
        if k % save_interval == 0:
            agent.save_param(k)
        if k % print_interval == 0:
            score_lst.append(total_score / print_interval)
            time_list.append(total_time / print_interval)
            #loss_list.append(loss.item() / print_interval)
            print(f"[{k}] : score[{total_score / print_interval}] time[{total_time / print_interval}] loss[{loss / print_interval}] eps[{eps}]")
            total_score = 0
            total_time = 0
            loss = 0
    graph_score(score_lst, time_list)

N = 50000
n = 256

#インポートでなく直接実行したときのみ処理を走らせるためのもの
if __name__ == "__main__":
    W = wrapper()
    aruco = arUco()
    C = Control()
    agent = Agent()
    main()