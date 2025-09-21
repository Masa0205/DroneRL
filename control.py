import airsim
import threading
import numpy as np
import os
import tempfile
import pprint
import cv2
import time

class Control:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.dis = 0.77
        self.vel = 1.0
        self.duration = self.dis / self.vel
        # AirSimクライアントへのアクセスを保護するためのロックを作成
        #self.airsim_lock = threading.Lock()
    def reset(self):
        #with self.airsim_lock:
            try:
                # シミュレーションリセット（物理状態リセット）
                self.client.reset()
            except BufferError as e:
                print("BufferError発生、再試行します:", e)
                time.sleep(1)
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self.client.reset()

            time.sleep(2.0)

            # 離陸して高度2.5mを初期位置とする
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync(4).join()

            # 位置 (x, y, z)
            pos = airsim.Vector3r(1.7, 0, -1.65)   # 注意: AirSimはz軸がマイナスで上方向
            # 姿勢 (回転)
            orn = airsim.Quaternionr(0, 0, 0, 1)  # 回転なし
            pose = airsim.Pose(pos, orn)
            self.client.simSetObjectPose("BP_Marker01_6", pose, True)
            time.sleep(0.5)
            
        

    def takeOff(self):
        self.client.takeoffAsync(2.5).join() #.joinはasyncなどの非同期処理完了を待つために必要

    def land(self):
        self.client.landAsync().join()

    def getImage(self):
        #with self.airsim_lock:
            responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if len(responses) == 0 or responses[0].height == 0 or responses[0].width == 0:
                print("画像が取得できませんでした")
                return None
            
            # バイト列からnumpy配列に変換
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            return img_rgb

    def hovering(self):
        #with self.airsim_lock:
            self.client.hoverAsync()
            time.sleep(self.duration)

    def front(self):
        #with self.airsim_lock:
            # 現在位置を取得
            pos = self.client.getMultirotorState().kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val

            # 相対距離 dx, dy, dz を指定して移動
            dx, dy, dz = self.dis, 0, 0
            # (x,y,z,v)
            self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel)  
            time.sleep(self.duration)


    def back(self):
        #with self.airsim_lock:
            # 現在位置を取得
            pos = self.client.getMultirotorState().kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val

            # 相対距離 dx, dy, dz を指定して移動
            dx, dy, dz = -self.dis, 0, 0
            self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel)  # 最後の1は速度[m/s]
            time.sleep(self.duration)

    def left(self):
        #with self.airsim_lock:
            # 現在位置を取得
            pos = self.client.getMultirotorState().kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val

            # 相対距離 dx, dy, dz を指定して移動
            dx, dy, dz = 0, -self.dis, 0
            self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel)  # 最後の1は速度[m/s]
            time.sleep(self.duration)

    def right(self):
        #with self.airsim_lock:
            # 現在位置を取得
            pos = self.client.getMultirotorState().kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val

            # 相対距離 dx, dy, dz を指定して移動
            dx, dy, dz = 0, self.dis, 0
            self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel)  # 最後の1は速度[m/s]
            time.sleep(self.duration)

    def up(self):
        # 現在位置を取得
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val

        # 相対距離 dx, dy, dz を指定して移動
        dx, dy, dz = 0, 0, -self.dis
        self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel) # 最後の1は速度[m/s]
        time.sleep(self.duration)

    def down(self):
        # 現在位置を取得
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val

        # 相対距離 dx, dy, dz を指定して移動
        dx, dy, dz = 0, 0, self.dis
        self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel) # 最後の1は速度[m/s]
        time.sleep(self.duration)

    def moveY(self, dy):
        # 現在位置を取得
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val

        # 相対距離 dx, dy, dz を指定して移動
        dx, dy, dz = 0, dy/50, 0
        self.client.moveToPositionAsync(x + dx, y + dy, z + dz, self.vel).join()  # 最後の1は速度[m/s]
        
    def test(self):
        # 現在位置を取得
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val
        print(f"({x}, {y}, {z})")
        # 相対距離 dx, dy, dz を指定して移動(0.77から挙動確認)
        dx, dy, dz = 0, 0.2, 0
        self.client.moveToPositionAsync(
            x + dx,
            y + dy,
            z + dz,
            self.vel,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),
            pos_tol = 0.01
        ).join()  # 最後の1は速度[m/s]
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val
        print(f"({x}, {y}, {z})")
    