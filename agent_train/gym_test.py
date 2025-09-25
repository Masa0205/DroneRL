import gymnasium as gym
import original_gym
import random
import numpy as np
import time

try:
    env = gym.make('DroneWorld-v0', render_mode="human")
    print("success")
except Exception as e:
    print("envError", e)

for i in range(10):

    obs ,_ = env.reset()
    while True:
        num = random.random()
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        action = np.array([x,y], dtype=np.float32)
        obs, r, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(1/30) # 30 FPS相当
        if terminated == True or truncated == True:
            break
env.close()
