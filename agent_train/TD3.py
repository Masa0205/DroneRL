import pickle
import random
import time
from collections import deque
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class replay_buffer(object):
    #N=バッファーサイズ, n=バッジサイズ
    def __init__(self, N, n):
        self.memory = deque(maxlen=N)
        self.n = n
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self):
        return random.sample(self.memory, self.n)
    
    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.device = device

        self.seq = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.seq.apply(self.init_weights)
    
    def forward(self, x):
        return self.seq(x) * self.action_max
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.seq = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.seq.apply(self.init_weights)
    
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.seq(x)
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
class TD3(object):
    def __init__(self):
        self.gamma = 0.99
        self.lr = 1e-3
        self.action_dim = 2
        self.state_dim = 4
        self.action_max = 1.0
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.t = 0
        self.tau = 0.005
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        print(self.device)
        #ActorNetwork
        self.actor = Actor(self.state_dim, self.action_dim, self.action_max, self.device).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_max, self.device).to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        #CriticNetwork-1
        self.c1 = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.target_c1 = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.optim_c1 = optim.Adam(self.c1.parameters(), lr=self.lr)
        #CriticNetwork-2
        self.c2 = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.target_c2 = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.optim_c2 = optim.Adam(self.c2.parameters(), lr=self.lr)

    
    def action(self, s, noise=0.0):
        s = torch.FloatTensor(s).to(self.device)
        a = self.actor(s)
        a = a.detach().cpu().numpy()  # Tensor → numpy
        if noise > 0.0:
            a += np.random.normal(0, noise*self.action_max,
                                       size=self.action_dim)
            action = np.clip(a, -self.action_max, self.action_max)
        #actionはnumpy配列になる[v(yaw), v(z)]
        return action
    
    def train(self, memory):
        self.t += 1
        s, a, r, s_prime, done = memory.sample()
         # Tensor化 & GPU転送
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(-1).to(self.device)
        s_prime = torch.FloatTensor(s_prime).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            noise = (
                torch.randn_like(a) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            #ターゲットActorで次状態の行動を推論
            a_prime = (
                self.target_actor(s_prime) + noise
            ).clamp(-self.action_max, self.action_max)

            #Q値取得
            target_q1 = self.target_c1(s_prime, a_prime)
            target_q2 = self.target_c2(s_prime, a_prime)
            #小さい方を選ぶ
            target_Q = torch.min(target_q1, target_q2)
            target_Q = r + target_Q * self.gamma * (1 - done)

        #現在のQ値取得
        q1 = self.c1(s, a)
        q2 = self.c2(s, a)

        #Criticの損失計算
        c1_loss = F.mse_loss(target_q1) + F.mse_loss(q1)
        c2_loss = F.mse_loss(target_q2) + F.mse_loss(q2)

        #Critic最適化
        self.optim_c1.zero_grad()
        c1_loss.backward()
        self.optim_c1.step()
        self.optim_c2.zero_grad()
        c2_loss.backward()
        self.optim_c2.step()

        #重みのコピーはソフト更新
        for param, c1target_param in zip(self.c1.parameters(), self.target_c1.parameters()):
            c1target_param.data.copy_(self.tau * param.data + (1 - self.tau) * c1target_param.data)
        for param, c2target_param in zip(self.c2.parameters(), self.target_c2.parameters()):
            c2target_param.data.copy_(self.tau * param.data + (1 - self.tau) * c2target_param.data)

        #2回に1回Actorも最適化
        if self.t % self.policy_freq == 0:
            #オンラインNNから行動を出力し、Criticが評価（「-」はCriticは最大化、Actorは最小化を目指すから）
            actor_loss = -self.c1(s, self.actor(s)).mean() #.mean()はバッチ平均
            
            #Actor最適化
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            # Update the frozen target models

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_param(self, k):
        torch.save(self.actor.state_dict(), f"actor_eps{k}.pth")
        torch.save(self.target_actor.state_dict(), f"actor_target_eps{k}.pth")
        torch.save(self.c1.state_dict(), f"c1_eps{k}.pth")
        torch.save(self.target_c1.state_dict(), f"c1_target_eps{k}.pth")
        torch.save(self.c2.state_dict(), f"c2_eps{k}.pth")
        torch.save(self.target_c2.state_dict(), f"c2_target_eps{k}.pth")

    def load(self, actor_path, c1_path, c2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.c1.load_state_dict(torch.load(c1_path))
        self.target_c1.load_state_dict(self.c1.state_dict())
        self.c2.load_state_dict(torch.load(c2_path))
        self.target_c2.load_state_dict(self.c2.state_dict())

        