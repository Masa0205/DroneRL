import gymnasium as gym
import original_gym
import random
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from TD3 import Replay_buffer, TD3

def get_state(obs, a):
    agent_pos = np.array(obs["agent"])
    target_pos = np.array(obs["target"])

    s_x, s_y = target_pos - agent_pos
    s_x, s_y = s_x/960, s_y/720
    state = np.array([s_x, s_y], dtype=np.float32)
    state = np.hstack((state,a))
    #print(state)
    return state

def graph_score(score_lst):
    now = datetime.now()
    date = now.date()
    x_axis = np.arange(len(score_lst))

    save_dir = "graphs"
    os.makedirs(save_dir, exist_ok=True)

    # 実験名＋タイムスタンプでファイル名を自動生成
    experiment_name = "gym_TD3"
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

def main():
    N = 50000
    n = 256
    memory = Replay_buffer(N, n)
    print_interval = 10
    save_interval = 1000
    score_lst = []
    total_score = 0.0
    loss = 0.0
    noise = 0.1
    episodes = 10000
    trigger = False
    p = 0
    for i in range(1,episodes+1):
        done = False
        obs ,_ = env.reset()
        a = np.array([0.0,0.0],dtype=np.float32)
        s = get_state(obs, a)
        while not done:
            if trigger:
                a = agent.action(s, noise)
            else:
                a = env.action_space.sample()
            #print(a)
            obs, r, terminated, truncated, _ = env.step(a)
            if terminated == True or truncated == True:
                done = True
            s_prime = get_state(obs, a)
            total_score += r
            memory.push((s, a, float(r), s_prime, int(done)))
            s = s_prime
            if len(memory) >= 2000:
                if p == 0:
                    print("学習開始")
                    p += 1
                trigger = True
                agent.train(memory)
            env.render()
            time.sleep(1/30) # 30 FPS相当
        if i % print_interval == 0:
            score_lst.append(total_score / print_interval)
            print(f"eps: {i}, r: {total_score / print_interval}")
            total_score = 0
        if i % save_interval == 0:
            agent.save_param(i)
    graph_score(score_lst)
    env.close()

if __name__ == "__main__":
    try:
        #見るときはrender_mode = "human"
        env = gym.make('DroneWorld-v0')
        print("success")
    except Exception as e:
        print("envError", e)
    agent = TD3()
    main()
    