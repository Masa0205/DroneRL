from djitellopy import Tello 
import time
import cv2 
import threading
from collections import deque
import queue
import pickle

from aruco import arUco
from TD3 import TD3

def image_thread(image_queue, action_queue):
    while True:
        try:
            # tellloから最新の画像を非同期で取得
            img = me.get_frame_read().frame
            if img is None:
                print("フレーム取得に失敗しました。スキップします。")
                time.sleep(0.1) # 少し待ってからリトライ
                continue # このループの残りの処理をスキップして次に進む
            stream_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # キューが満杯なら一番古い画像を破棄
            if image_queue.full():
                image_queue.get()
            # 最新の画像をキューに追加
            image_queue.put(output_img)
            #print("success")

            #状態取得
            a = [0.0, 0.0]
            centers, are, output_img = aruco.detect(stream_bgr)
            if len(centers) > 0:
                s = aruco.get_state(centers, a)
                a = agent.action(s)
            else:
                a = [0.0, 0.0]
            
            # キューが満杯なら一番古い画像を破棄
            if action_queue.full():
                action_queue.get()
            # 最新の画像をキューに追加
            action_queue.put(a)

            
            




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
        


def main():
    print(me.get_battery())
    image_queue = queue.Queue(maxsize=4) # 4フレームを保持
    action_queue = queue.Queue(maxsize=1)
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
    agent = TD3()
    aruco = arUco()
    try :
        agent.load("actor_target_esp10000", "critic_target_esp10000")
        print("ロード成功")
    except Exception as e:
        print("モデル読み込みエラー",e)

    main()