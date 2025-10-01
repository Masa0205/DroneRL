from djitellopy import Tello 
import time
import cv2   
import threading
from collections import deque
import queue
import numpy as np
from aruco import arUco
from TD3 import TD3


def image_thread(image_queue):
    scale = 30
    area = None
    while True:
        try:

            # tellloから最新の画像を非同期で取得
            img = me.get_frame_read().frame
            if img is None:
                print("フレーム取得に失敗しました。スキップします。")
                time.sleep(0.1) # 少し待ってからリトライ
                continue # このループの残りの処理をスキップして次に進む
            stream_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            centers, area, output_img = aruco.detect(stream_bgr)
            # キューが満杯なら一番古い画像を破棄
            if image_queue.full():
                image_queue.get()
            # 最新の画像をキューに追加
            image_queue.put(output_img)
            #print("success")

            #状態取得
            norm_a = np.array([0.0, 0.0], dtype=np.float32)
            #print("area=",area)
            MARGIN = 50
            if len(centers) > 0:
                x, y = centers
                if (480 - MARGIN) < x < (480+MARGIN) and (360-MARGIN) < y < (360+MARGIN):
                    a = norm_a
                    #print("Done")
                else:
                    s = aruco.get_state(centers, a)
                    a = agent.action(s)
                    print("Tracking")
            else:
                a = norm_a
                #print("Nodetect")
            yaw, z = a
            yaw = - int(scale * yaw)
            z = int(scale * z)
            print(f"yaw={yaw}, z={z}")
            me.send_rc_control(0, 0, z, yaw)
            # 取得間隔を調整（例: 0.03秒 = 約33fps）
            time.sleep(1/30)
        except Exception as e:
            print("Error:", e)
            
        
def main():
    print(me.get_battery())
    me.takeoff()
    time.sleep(1)
    image_queue = queue.Queue(maxsize=1) # 4フレームを保持
    thread = threading.Thread(target=image_thread, args=(image_queue,))
    thread.daemon = True # メインスレッド終了時にスレッドも終了
    thread.start()
    print("thread on")
    while True:
        
        if not image_queue.empty():
            frame = image_queue.get()
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
        agent.load("actor_target_eps10000.pth", "critic_target_eps10000.pth")
        print("ロード成功")
    except Exception as e:
        print("モデル読み込みエラー",e)

    main()