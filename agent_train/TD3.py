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

class Replay_buffer(object):
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
        self.seqc1 = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.seqc2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.seqc1.apply(self.init_weights)
        self.seqc2.apply(self.init_weights)
    
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.seqc1(x), self.seqc2(x)
    
    def Q1(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.seqc1(x)
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
class TD3(object):
    def __init__(self):
        self.gamma = 0.99
        self.lr = 3e-4
        self.action_dim = 2
        self.state_dim = 4
        self.action_max = 1.0
        self.policy_noise = 0.1
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.t = 0
        self.tau = 0.01
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
        #CriticNetwork
        self.critic = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.device).to(self.device)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    
    def action(self, s, noise=0.0):
        s = torch.FloatTensor(s).to(self.device)
        a = self.actor(s)
        a = a.detach().cpu().numpy()  # Tensor → numpy
        if noise > 0.0:
            #print("noiseOn")
            a += np.random.normal(0, noise*self.action_max,
                                       size=self.action_dim)
        action = np.clip(a, -self.action_max, self.action_max)
        #actionはnumpy配列になる[v(yaw), v(z)]
        return action
    
    def train(self, memory):
        self.t += 1
        batch = memory.sample()
        s, a, r, s_prime, done = zip(*batch)

         # Tensor化 & GPU転送
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.FloatTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(-1).to(self.device)
        s_prime = torch.FloatTensor(np.array(s_prime)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            noise = (
                torch.randn_like(a) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            #ターゲットActorで次状態の行動を推論
            a_prime = (
                self.target_actor(s_prime) + noise
            ).clamp(-self.action_max, self.action_max)

            #Q値取得
            target_q1, target_q2 = self.target_critic(s_prime, a_prime)
            
            #小さい方を選ぶ
            target_Q = torch.min(target_q1, target_q2)
            target_Q = r + target_Q * self.gamma * (1 - done)

        #現在のQ値取得
        q1, q2 = self.critic(s, a)
       

        #Criticの損失計算
        c_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        #Critic最適化
        self.optim_critic.zero_grad()
        c_loss.backward()
        self.optim_critic.step()
        

        #重みのコピーはソフト更新
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

        #2回に1回Actorも最適化
        if self.t % self.policy_freq == 0:
            #オンラインNNから行動を出力し、Criticが評価（「-」はCriticは最大化、Actorは最小化を目指すから）
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean() #.mean()はバッチ平均
            
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
        torch.save(self.critic.state_dict(), f"critic_eps{k}.pth")
        torch.save(self.target_critic.state_dict(), f"critic_target_eps{k}.pth")
        

    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path, weights_only=True, map_location=torch.device(self.device)))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(critic_path, weights_only=True, map_location=torch.device(self.device)))
        self.target_critic.load_state_dict(self.critic.state_dict())
        

        