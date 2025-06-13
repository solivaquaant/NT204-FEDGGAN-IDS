import socket
import pickle
import struct
import threading
import copy
import numpy as np
import time
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

HOST = '0.0.0.0'
PORT = 8888
NUM_CLIENTS = 1

# Global dictionary to store received weights from clients
global_gan_models = {}  # input_size: (gen, disc)
received_weights = {}   # input_size: [list các update]
lock = threading.Lock()
learning_rate = 0.0002

# GAN global
def build_global_models(input_size): 
    class DiscriminatorNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)
    class GeneratorNet(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(5, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, output_size),
                nn.Tanh()
            )
        def forward(self, x):
            return self.model(x)
    return GeneratorNet(input_size), DiscriminatorNet(input_size)

# Cấu hình tham số cho các client 
def get_default_config():
    # Trả về dict các tham số cấu hình cho client
    return {
        'gan_epochs': 10,
        'ids_epochs': 10,
        'batch_size': 1024,
        'num_samples_normal': 10,
        'num_samples_weak': 10,
        'lr_disc': 0.0002,
        'lr_gen': 0.0002,
        'penalty_coef': 0.1,
        'labels_to_consider': 15,
        'f1_threshold': 0.9,
       #'global_epochs': 5, # Cần sửa hàm run_global_training (client.py) để chạy

    }

def handle_client(conn, addr, client_id):
    """Handle a client connection."""
    print(f"[+] Client {client_id} connected from {addr}")
    try:
        # 1. Gửi config cho client-----------------
        # Client tự chọn bộ dữ liệu

        config_dict = get_default_config() # Cấu hình tham số toàn cục
        send_data(conn, config_dict) # Gửi đến client
        print(f"[=] Sent config to Client {client_id}")
        
        # 2. Nhận local GAN weights từ client--------------
        data = receive_data(conn) #Nhận dữ liệu
        input_size = data.get('input_size') #Lấy input-size

        print(f"[=] Received local GAN weights from Client {client_id} (input_size={input_size})")
        with lock:
            # Nếu input_size mới 
            if input_size not in global_gan_models: #Xây dựng model GAN global ứng với input size mới
                global_gan_models[input_size] = build_global_models(input_size) 
            if input_size not in received_weights: #Khởi tạo danh sách chứa trọng số mới dành cho input_size
                received_weights[input_size] = []

            # Thêm trọng số vào danh sách theo input_size
            # Các client có input_size là 42, sẽ chung một danh sách trọng số
            # client có input_size là 45, sẽ chung một danh sách trọng số    
            received_weights[input_size].append(data) 

        # 3. Đợi các client khác (nếu cần)------------------------
        wait_start = time.time()
        while sum(len(lst) for lst in received_weights.values()) < NUM_CLIENTS:
            time.sleep(0.1)
            if time.time() - wait_start > 60:
                print(f"[!] Timeout waiting for all clients. Proceeding with {sum(len(lst) for lst in received_weights.values())} clients.")
                break
        
        # 4. Federated averaging cho Generator và Discriminator -----------------
        with lock:
            #Lặp qua từng trọng số của client theo input_size
            for isize, updates in received_weights.items(): 
                gen, disc = global_gan_models[isize] #Lấy model G, D toàn cục
                gen_states = [c['gen_state'] for c in updates] #Danh sách grads của các G_l client
                disc_states = [c['disc_state'] for c in updates] #Danh sách grads của các D_l client
                
                # Tính trung bình Generator
                gen_avg = {}
                for key in gen.state_dict().keys(): # Duyệt qua tên layer trong model
                    # Tính trung bình các tensor trọng số
                    gen_avg[key] = torch.stack([torch.tensor(gs[key]) for gs in gen_states], 0).float().mean(0)
                gen.load_state_dict(gen_avg) #Nạp vào Generator Global

                # Trung bình Discriminator, tương tự Generator
                disc_avg = {}
                for key in disc.state_dict().keys():
                    disc_avg[key] = torch.stack([torch.tensor(ds[key]) for ds in disc_states], 0).float().mean(0)
                disc.load_state_dict(disc_avg)
        
        # 5. Gửi lại weights mới cho client ----------------------------
        gen, disc = global_gan_models[input_size] # input_size lấy từ client
        # Đóng gói dữ liệu
        package = {
            'new_gen_weights': gen.state_dict(),
            'new_disc_weights': disc.state_dict()
        }
        # Gửi dữ liệu
        send_data(conn, package)
        print(f"[✔] Sent new GAN weights to Client {client_id}")
        time.sleep(1)
    except Exception as e:
        print(f"[!] ❌ Error with Client {client_id}: {e}")
        traceback.print_exc()
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except:
            pass
        conn.close()
        print(f"[x] Connection with Client {client_id} closed")

# Hàm tương tác vói client -----------------------
# Gửi dữ liệu
def send_data(conn, data):
    """Send serialized data over the connection."""
    try:
        serialized = pickle.dumps(data)
        # Send data length as 4 bytes
        conn.sendall(struct.pack('!I', len(serialized)))
        # Send the actual data
        conn.sendall(serialized)
        print(f"Sent {len(serialized)} bytes")
        return True
    except Exception as e:
        print(f"Error sending data: {e}")
        return False

#Nhận dữ liệu
def receive_data(conn):
    """Receive serialized data from the connection."""
    try:
        # Receive 4 bytes for data length
        raw_length = conn.recv(4)
        if not raw_length:
            raise ConnectionError("Client closed connection (length)")
        length = struct.unpack('!I', raw_length)[0]
        print(f"Expecting to receive {length} bytes")
        
        # Receive the actual data
        data = b""
        while len(data) < length:
            packet = conn.recv(min(4096, length - len(data)))
            if not packet:
                raise ConnectionError(f"Incomplete data received: {len(data)}/{length} bytes")
            data += packet
            print(f"Received {len(data)}/{length} bytes")
        
        # Deserialize the data
        return pickle.loads(data)
    except Exception as e:
        print(f"Error receiving data: {e}")
        traceback.print_exc()
        raise

#------------MAIN--------------
def main(): 
    # xóa trọng số cũ
    received_weights.clear()
    
    # Khởi tạo model toàn cục
    global global_gan_models
    current_client = 0

    # Tạo socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try: 
        # Lắng nghe trên port 8888
        server_socket.bind((HOST, PORT))
        server_socket.listen(NUM_CLIENTS)
        print(f"🚀 Server listening at {HOST}:{PORT}...")
        # Tạo luồng
        threads = []
        
        # Accept connections from clients
        while current_client < NUM_CLIENTS:
            try:
                # Tạo luồng
                conn, addr = server_socket.accept() 
                # Chạy hàm xử lí client 
                client_thread = threading.Thread(target=handle_client, args=(conn, addr, current_client))
                client_thread.daemon = True  # Set as daemon so it doesn't block program exit
                client_thread.start()
                threads.append(client_thread)
                current_client += 1
            except KeyboardInterrupt:
                print("\nServer shutdown requested...")
                break
            except Exception as e:
                print(f"Error accepting client connection: {e}")
                continue
        
        # Wait for all client threads to complete
        for thread in threads:
            thread.join()
        print("All clients processed. Server shutting down.")

    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()

    finally:
        server_socket.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer terminated by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()