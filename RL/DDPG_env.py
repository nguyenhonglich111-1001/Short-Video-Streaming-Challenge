import numpy as np

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
            # Buffer size của mỗi player
            "buffer_size": [player.get_buffer_size() for player in self.players],
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
        # Cập nhật K với giới hạn tối thiểu
        self.K = max(0.01, self.K + K_adjust)
        # Cập nhật P_ZERO với giới hạn tối thiểu
        self.P_ZERO = max(0.01, self.P_ZERO + P_ZERO_adjust)

        # Giả lập quá trình dự đoán băng thông và tải xuống video
        # (cần tích hợp với mã chính của bạn)
        for player in self.players:
            # Cập nhật băng thông và buffer size từ player
            bandwidth = player.get_bandwidth()  # Hàm giả lập băng thông
            self.state["bandwidth"].append(bandwidth)
            self.state["buffer_size"][player.id] = player.get_buffer_size()

            # Tính toán MACD mới
            ema_short = Algorithm._ewma(
                np.array(self.state["bandwidth"]), EMA_SHORT_WINDOW)[-1]
            ema_long = Algorithm._ewma(
                np.array(self.state["bandwidth"]), EMA_LONG_WINDOW)[-1]
            self.state["macd"] = ema_short - ema_long

        # Tính reward dựa trên hiệu quả của K và P_ZERO
        self.reward = self.calculate_reward()

        # Kiểm tra nếu tất cả các video đã được tải hết
        self.done = all(player.get_remain_video_num()
                        == 0 for player in self.players)

        return self.state, self.reward, self.done

    def calculate_reward(self):
        """
        Tính reward dựa trên:
        - Giảm thiểu rebuffering.
        - Tăng chất lượng video (bitrate cao hơn).
        """
        rebuffer_penalty = sum(player.get_rebuffer_time()
                               for player in self.players)
        avg_bitrate = np.mean([np.mean(player.get_downloaded_bitrate())
                              for player in self.players if player.get_downloaded_bitrate()])

        # Reward là giá trị bitrate trung bình trừ đi penalty do rebuffering
        return avg_bitrate - 10 * rebuffer_penalty
