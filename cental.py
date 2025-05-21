import copy
import numpy as np

# Giả lập mô hình Discriminator và Generator
class Generator:
    def __init__(self, name="G"):
        self.weights = np.random.randn(10)  # vector 10 chiều giả lập

    def train(self, data, epochs=1):
        self.weights += np.random.randn(10) * 0.01  # Giả lập training

class Discriminator:
    def __init__(self, name="D"):
        self.weights = np.random.randn(10)

    def train(self, data, epochs=1):
        self.weights += np.random.randn(10) * 0.01

# Tham số ban đầu
E = 15                # Số global epochs
K = 5                   # Khoảng đồng bộ
M = 2                   # Số client
client_data = [np.random.randn(100, 10) for _ in range(M)]
P = [1.0 / M] * M       # Trọng số của mỗi client

# Khởi tạo tham số toàn cục
global_G = Generator("Global_G")
global_D = Discriminator("Global_D")

w_t = global_G.weights
θ_t = global_D.weights

# Bắt đầu training
for t in range(1, E + 1):
    print(f"Epoch {t}...")

    local_w = []
    local_θ = []

    for n in range(M):  # Vòng qua từng client
        local_G = copy.deepcopy(global_G)
        local_D = copy.deepcopy(global_D)

        local_G.train(client_data[n], epochs=1)
        local_D.train(client_data[n], epochs=1)

        local_w.append(local_G.weights)
        local_θ.append(local_D.weights)

    # Đồng bộ hóa mỗi K epoch
    if t % K == 0:
        # Tổng trọng số có trọng số P
        w_t = sum(P[n] * local_w[n] for n in range(M))
        θ_t = sum(P[n] * local_θ[n] for n in range(M))

        global_G.weights = w_t
        global_D.weights = θ_t

        print(f"→ Updated global weights at epoch {t}")

# Kết quả cuối cùng
print("✅ Training hoàn tất.")
print("Final Generator weights:", global_G.weights)
print("Final Discriminator weights:", global_D.weights)
