from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from enum import Enum
class DroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.width = 960
        self.height = 720
        self.REAL_WIDTH_CM = 400
        self.REAL_HEIGHT_CM = 300
        # 2. スケールは自動的に再計算される (結果は 2.4)
        self.PX_PER_CM = self.width / self.REAL_WIDTH_CM
        # Telloの最大速度 (cm/s)
        self.MAX_SPEED_CM_S = 100.0
        self.MARGIN = 50
        self.STEP_LIMIT = 50
        self.dt = 1/30
        self.step_count = 1
        self.action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
        self.agent_location = np.array([-1.0, -1.0], dtype=np.float32)
        self.target_location = np.array([480, 360], dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(low=np.array([0,0]),high=np.array([self.width, self.height]), shape=(2,), dtype=np.float32),   # [x, y] coordinates
                "target": gym.spaces.Box(low=np.array([0,0]),high=np.array([self.width, self.height]), shape=(2,), dtype=np.float32)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}
    def get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent_location = self.target_location
        while np.array_equal(self.target_location, self.agent_location):
            self.agent_location = self.np_random.uniform(np.array([0,0]), np.array([self.width, self.height]), size=(2,)).astype(np.float32)

        observation = self.get_obs()
        info = self.get_info()
        if self.render_mode == "human":
            self.render_frame()
        self.step_count = 1
        return observation, info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        #送られてきたアクションから1フレームで何ピクセル動くかに変換
        scaled_action = action*self.MAX_SPEED_CM_S
        action_cmd = scaled_action.astype(np.int32)
        action_sim = action_cmd * self.PX_PER_CM
        
        dis = action_sim*self.dt
        self.agent_location += dis
        #Doneチェック
        x,y = self.agent_location
        if not (0 < x < self.width and 0 < y < self.height):
            terminated = True
            reward -= 1
        elif (480 - self.MARGIN) < x < (480+self.MARGIN) and (360-self.MARGIN) < y < (360+self.MARGIN):
            terminated = True
            reward += 10
        else:
            if self.step_count == 50:
                truncated = True
                reward = -1
            else:
                cx,cy = [480, 360]
                dx, dy = cx - x, cy - y
                distance = np.sqrt(dx**2 + dy**2)
                d_max = np.sqrt((cx)**2 + (cy)**2)
                # 正規化
                norm_dist = distance / d_max
                reward = 1 - norm_dist
        
        observation = self.get_obs()
        info = self.get_info()
        if self.render_mode == "human":
            self.render_frame()
        self.step_count += 1
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()
    
    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.width, self.height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))
        

        # First we draw the target
        x,y = self.target_location
        top_left_x = x - self.MARGIN
        top_left_y = y - self.MARGIN
        box_width = self.MARGIN *2
        box_height = self.MARGIN *2
        target_rect = pygame.Rect(top_left_x, top_left_y, box_width,
                                  box_height)
        pygame.draw.rect(
            canvas,            # 描画対象
            (0,255,0), # 色
            target_rect,       # 四角形の位置とサイズ
        )
            
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.agent_location,
            10,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()






        





        
