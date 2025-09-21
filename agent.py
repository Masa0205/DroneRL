import pickle
import random
import time
from collections import deque

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

class model(nn.Module):
    #n_frame未定義おそらく画像加工後のサイズ←フレーム数でした
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)
        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(self.init_weights)
    
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))

        return q
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
class Agent:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 1e-4
        self.action_size = 3
        self.n_frame = 4
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        print(self.device)
        self.q = model(self.n_frame, self.action_size, self.device).to(self.device)
        self.target_q = model(self.n_frame, self.action_size, self.device).to(self.device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        print(self.device)
    
    def action(self, s, eps):
        action_space = {0: "right",
                        1: "left",
                        2: "hovering"}

        if eps > np.random.rand():
            # ランダム行動
            a = np.random.choice(list(action_space.keys()))
        else:
            # モデルのデバイスに移動
            s = s.to(self.device)

            # 推論
            q_values = self.q(s)

            # CPU に戻して NumPy 配列に変換
            a = np.argmax(q_values.cpu().detach().numpy())

        return a
    #終端判定Doneの定義方法
    def train(self, memory):
        s, r, a, s_prime, done = list(map(list, zip(*memory.sample())))
        s = np.array(s).squeeze()
        s_prime = np.array(s_prime).squeeze()
        a_max = self.q(s_prime).max(1)[1].unsqueeze(-1)
        r = torch.FloatTensor(r).unsqueeze(-1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            y = r + self.gamma * self.target_q(s_prime).gather(1, a_max) * done
        a = torch.tensor(a).unsqueeze(-1).to(self.device)
        q_value = torch.gather(self.q(s), dim=1, index=a.view(-1, 1).long())
        #self.q(s)の意味確認
        loss = F.smooth_l1_loss(q_value, y).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def copy_weights(self):
        q_dict = self.q.state_dict()
        self.target_q.load_state_dict(q_dict)

    def save_param(self, k):
        torch.save(self.q.state_dict(), f"drone_eps{k}.pth")
        torch.save(self.target_q.state_dict(), f"drone_target_eps{k}.pth")

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.target_q.load_state_dict(self.q.state_dict())

        