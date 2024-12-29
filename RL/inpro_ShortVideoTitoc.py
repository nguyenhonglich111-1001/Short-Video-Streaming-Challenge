#1. Thiết lập môi trường (Environment)
import numpy as np
from simulator.video_player import Player

class VideoEnv:
    def __init__(self, players, initial_K=0.5, initial_P_ZERO=0.1):
        self.players = players  # Danh sách các trình phát video
        self.K = initial_K  # Tham số K ban đầu
        self.P_ZERO = initial_P_ZERO  # Tham số P_ZERO ban đầu
        self.state = None  # Trạng thái hiện tại của môi trường
        self.reward = 0  # Reward sau mỗi hành động
        self.done = False  # Đánh dấu khi quá trình tải video kết thúc

    def reset(self):
        # Khởi tạo lại môi trường
        self.state = {
            "bandwidth": [],  # Lịch sử băng thông
            "buffer_size": [player.get_buffer_size() for player in self.players],  # Buffer size của mỗi player
            "macd": 0.0,  # Giá trị MACD
            "future_bandwidth": 0.0  # Băng thông dự đoán
        }
        self.K = 0.5
        self.P_ZERO = 0.1
        self.reward = 0
        self.done = False
        return self.state

    def step(self, action):
        """
        Thực hiện một hành động (action) để cập nhật K và P_ZERO.
        Action sẽ là một vector [K_adjust, P_ZERO_adjust].
        """
        K_adjust, P_ZERO_adjust = action
        self.K = max(0.01, self.K + K_adjust)  # Cập nhật K với giới hạn tối thiểu
        self.P_ZERO = max(0.01, self.P_ZERO + P_ZERO_adjust)  # Cập nhật P_ZERO với giới hạn tối thiểu

        # Giả lập quá trình dự đoán băng thông và tải xuống video
        # (cần tích hợp với mã chính của bạn)
        for player in self.players:
            # Cập nhật băng thông và buffer size từ player
            bandwidth = player.get_bandwidth()  # Hàm giả lập băng thông
            self.state["bandwidth"].append(bandwidth)
            self.state["buffer_size"][player.id] = player.get_buffer_size()

            # Tính toán MACD mới
            ema_short = Algorithm._ewma(np.array(self.state["bandwidth"]), EMA_SHORT_WINDOW)[-1]
            ema_long = Algorithm._ewma(np.array(self.state["bandwidth"]), EMA_LONG_WINDOW)[-1]
            self.state["macd"] = ema_short - ema_long

        # Tính reward dựa trên hiệu quả của K và P_ZERO
        self.reward = self.calculate_reward()

        # Kiểm tra nếu tất cả các video đã được tải hết
        self.done = all(player.get_remain_video_num() == 0 for player in self.players)
        
        return self.state, self.reward, self.done

    def calculate_reward(self):
        """
        Tính reward dựa trên:
        - Giảm thiểu rebuffering.
        - Tăng chất lượng video (bitrate cao hơn).
        """
        rebuffer_penalty = sum(player.get_rebuffer_time() for player in self.players)
        avg_bitrate = np.mean([np.mean(player.get_downloaded_bitrate()) for player in self.players if player.get_downloaded_bitrate()])

        # Reward là giá trị bitrate trung bình trừ đi penalty do rebuffering
        return avg_bitrate - 10 * rebuffer_penalty

#2. Deep Deterministic Policy Gradient (DDPG)
import torch
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

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