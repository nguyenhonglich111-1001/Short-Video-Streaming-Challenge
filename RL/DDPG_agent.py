
#2. Deep Deterministic Policy Gradient (DDPG)
# import torch
# import gym
import sys
sys.path.append(
    'D:\\Future_Internet_Lab\\Short-Video-Streaming-Challenge')


from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from DDPG_env import VideoEnv
from simulator.video_player import Player
import numpy as np

PLAYER_NUM = 5

# Tạo môi trường
env = VideoEnv(players=[Player() for _ in range(PLAYER_NUM)])  # Khởi tạo danh sách các Player
env = DummyVecEnv([lambda: env])  # Vector hóa môi trường để sử dụng với DDPG

# Định nghĩa noise cho không gian hành động
n_actions = 2  # [K_adjust, P_ZERO_adjust]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Khởi tạo DDPG model
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    batch_size=128,
    gamma=0.99,  # Discount factor
    tau=0.005,  # Soft update
    train_freq=(1, "episode"),
    gradient_steps=1,
    device="auto"
)

# Callback để lưu mô hình
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ddpg_video')

# Huấn luyện mô hình
model.learn(total_timesteps=50000, callback=checkpoint_callback)

# Lưu mô hình
model.save("final_ddpg_model")

#3. Sau khi huấn luyện mô hình DDPG, chúng ta có thể sử dụng mô hình này để điều chỉnh K và P_ZERO trong thuật toán bằng mã code
from stable_baselines3 import DDPG

# Load model
trained_model = DDPG.load("final_ddpg_model")

# Sử dụng mô hình để điều chỉnh tham số
state = env.reset()
done = False

while not done:
    # Lấy hành động từ mô hình
    action, _ = trained_model.predict(state, deterministic=True)

    # Cập nhật môi trường
    state, reward, done = env.step(action)

    # Cập nhật thuật toán chính với K và P_ZERO mới
    Algorithm.K = env.K
    Algorithm.P_ZERO = env.P_ZERO