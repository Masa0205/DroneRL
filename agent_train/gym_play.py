import gymnasium as gym
import original_gym
import random
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from TD3 import Replay_buffer, Actor, Critic, TD3

def get_state(obs, a):
    agent_pos = np.array(obs["agent"])
    target_pos = np.array(obs["target"])

    state = (target_pos - agent_pos).astype(np.float32)
    state = np.hstack((state,a))
    #print(state)
    return state


def main():
    N = 50000
    n = 256
    memory = Replay_buffer(N, n)
    print_interval = 10
    save_interval = 2000
    score_lst = []
    total_score = 0.0
    loss = 0.0
    noise = 0.0
    episodes = 10
    trigger = False
    p = 0
    for i in range(1,episodes+1):
        done = False
        obs ,_ = env.reset()
        a = np.array([0.0,0.0],dtype=np.float32)
        s = get_state(obs, a)
        while not done:
            a = agent.action(s, noise)
            print(a)
            obs, r, terminated, truncated, _ = env.step(a)
            if terminated == True or truncated == True:
                done = True
            s_prime = get_state(obs, a)
            s = s_prime
            env.render()
            time.sleep(1/30) # 30 FPS相当

    env.close()

if __name__ == "__main__":
    try:
        #見るときはrender_mode = "human"
        env = gym.make('DroneWorld-v0', render_mode="human")
        print("success")
    except Exception as e:
        print("envError", e)
    agent = TD3()
    agent.load("actor_target_eps10000.pth","critic_target_eps10000.pth")
    main()
    