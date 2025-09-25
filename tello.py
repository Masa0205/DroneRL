from djitellopy import Tello
import pickle
import random
import time
import cv2
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
from agent import Agent
from img import arUco, wrapper


def image_thread(image_queue, view_queue):
    while True:
        try:
            # tellloから最新の画像を非同期で取得
            img = me.get_frame_read().frame
            if img is None:
                print("フレーム取得に失敗しました。スキップします。")
                time.sleep(0.1) # 少し待ってからリトライ
                continue # このループの残りの処理をスキップして次に進む
            stream_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #cv2.imshow("Image", stream_rgb)
            #cv2.waitKey(1)
            # 画像データをnumpy配列に変換
            
            # キューが満杯なら一番古い画像を破棄
            if image_queue.full():
                image_queue.get()
            if view_queue.full():
                view_queue.get()
            # 最新の画像をキューに追加
            image_queue.put(stream_rgb)
            view_queue.put(stream_rgb)
            #print("success")
            # 取得間隔を調整（例: 0.03秒 = 約33fps）
            time.sleep(1/30)
        except Exception as e:
            print("getImageError", e)
            
        
def flight(image_queue):
    dis = 80
    vel = 100
    duration = dis / vel
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(3)
    print("start")
    while True:
        
        local_images = clip(image_queue, [])
        check, centers = marker_check(local_images)
        #print(check, centers)
        
        if check:
            s = W.preprocess(local_images)
            a = agent.action(s, 0)
            if a == 0:
                me.send_rc_control(0, 0, 0, 0)
                #print("a=",a)
            elif a == 1:
                me.send_rc_control(-vel, 0, 0, 0)
                #print("a=",a)
            elif a == 2:
                me.send_rc_control(vel, 0, 0, 0)
                #print("a=",a)
            
        else:
            me.send_rc_control(0, 0, 0, 0)
            #print("noDetect")
        
        time.sleep(duration)
        
def clip(image_queue, local_images):
    while not image_queue.empty():
            try:
                img = image_queue.get_nowait()
                local_images.append(img)
            except queue.Empty:
                # 例外処理: 複数のスレッドが同時にアクセスした場合の安全対策
                break
    return local_images

def marker_check(local_images):
    check = False
    centers = []
    for img in reversed(local_images):
        try:
            centers = aruco.detect(img)
            if len(centers) > 0:
                check = True
                break
        except Exception as e:
            print("detectError:", e)
    return check, centers

def main():
    print(me.get_battery())
    image_queue = queue.Queue(maxsize=4) # 4フレームを保持
    view_queue = queue.Queue(maxsize=1)
    local_images = []
    thread_1 = threading.Thread(target=image_thread, args=(image_queue,view_queue,))
    thread_1.daemon = True # メインスレッド終了時にスレッドも終了
    thread_1.start()
    me.takeoff()
    thread_2 = threading.Thread(target=flight, args=(image_queue,))
    thread_2.daemon = True
    thread_2.start()
    while True:
        if not view_queue.empty():
            frame = view_queue.get()
            cv2.imshow("Tello", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    me.land()
    me.streamoff()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    me = Tello()
    me.connect()
    me.streamon()
    agent = Agent()
    aruco = arUco()
    W = wrapper()
    try :
        agent.load("model_2.pth")
        print("ロード成功")
    except Exception as e:
        print("モデル読み込みエラー",e)

    main()