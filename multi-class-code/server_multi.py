import socket
import pickle
import struct
import threading
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import traceback
from collections import defaultdict

# Global variables
lock = threading.Lock()
global_gan_models = {}  # {input_size: (generator, discriminator)}
global_ids_models = {}  # {input_size: ids_model}
received_weights = defaultdict(list)  # {input_size: [client_weights]}


def build_global_models(input_size):
    """
    Create global Generator and Discriminator models.

    Args:
        input_size (int): Feature input size.

    Returns:
        tuple: (GeneratorNet, DiscriminatorNet)
    """
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

    return GeneratorNet(input_size), DiscriminatorNet(input_size)


def build_global_ids_model(input_size, num_classes=2):
    """
    Create global IDS model (CNNClassifier).

    Args:
        input_size (int): Feature input size.
        num_classes (int): Number of classes.

    Returns:
        nn.Module: CNNClassifier
    """
    class CNNClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
            self.num_classes = num_classes
            self.softmax = nn.Softmax(
                dim=1) if num_classes > 2 else nn.Sigmoid()

        def forward(self, x):
            logits = self.model(x)
            return self.softmax(logits) if self.num_classes > 2 else logits
    return CNNClassifier(input_size, num_classes)


def send_data(conn, data):
    """
    Send serialized data to client.

    Args:
        conn (socket.socket): Connection socket.
        data: Data to send.
    """
    try:
        serialized = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(serialized)))
        conn.sendall(serialized)
        print(f"[=] Sent {len(serialized)} bytes")
    except Exception as e:
        print(f"[!] Error sending data: {e}")
        raise


def receive_data(conn):
    """
    Receive serialized data from client.

    Args:
        conn (socket.socket): Connection socket.

    Returns:
        Data received.
    """
    try:
        raw_length = conn.recv(4)
        if not raw_length:
            raise ConnectionError("Client closed connection")
        length = struct.unpack('!I', raw_length)[0]
        print(f"[=] Expecting to receive {length} bytes")

        data = b""
        while len(data) < length:
            packet = conn.recv(min(4096, length - len(data)))
            if not packet:
                raise ConnectionError(
                    f"Incomplete data: {len(data)}/{length} bytes")
            data += packet
            print(f"[=] Received {len(data)}/{length} bytes")

        return pickle.loads(data)
    except Exception as e:
        print(f"[!] Error receiving data: {e}")
        raise


def handle_client(conn, addr, client_id, config):
    """Handle a client connection."""
    print(f"[+] Client {client_id} connected from {addr}")
    try:
        config_dict = copy.deepcopy(config)
        send_data(conn, config_dict)
        print(f"[=] Sent config to Client {client_id}")

        data = receive_data(conn)
        input_size = data.get('input_size')
        gen_state = data.get('gen_state')
        disc_state = data.get('disc_state')
        ids_state = data.get('ids_state', None)
        num_classes = data.get('num_classes', 2)  # Default to binary

        if input_size is None or gen_state is None or disc_state is None:
            raise ValueError(f"Invalid data received from Client {client_id}")

        print(
            f"[=] Received local weights from Client {client_id} (input_size={input_size}, num_classes={num_classes})")
        with lock:
            if input_size not in global_gan_models:
                global_gan_models[input_size] = build_global_models(input_size)
            if input_size not in global_ids_models and ids_state is not None:
                global_ids_models[input_size] = build_global_ids_model(
                    input_size, num_classes)

            received_weights[input_size].append({
                'gen_state': gen_state,
                'disc_state': disc_state,
                'ids_state': ids_state
            })

        wait_start = time.time()
        while sum(len(lst) for lst in received_weights.values()) < config['num_clients']:
            time.sleep(0.1)
            if time.time() - wait_start > config['timeout']:
                print(
                    f"[!] Timeout waiting for clients. Proceeding with {sum(len(lst) for lst in received_weights.values())} clients.")
                break

        with lock:
            for isize, updates in received_weights.items():
                gen, disc = global_gan_models[isize]
                ids = global_ids_models.get(isize)

                gen_states = [c['gen_state'] for c in updates]
                gen_avg = {}
                for key in gen.state_dict().keys():
                    tensors = [torch.tensor(gs[key], dtype=torch.float32)
                               for gs in gen_states]
                    gen_avg[key] = torch.stack(tensors, 0).mean(0)
                gen.load_state_dict(gen_avg)

                disc_states = [c['disc_state'] for c in updates]
                disc_avg = {}
                for key in disc.state_dict().keys():
                    tensors = [torch.tensor(ds[key], dtype=torch.float32)
                               for ds in disc_states]
                    disc_avg[key] = torch.stack(tensors, 0).mean(0)
                disc.load_state_dict(disc_avg)

                if ids is not None:
                    ids_states = [c['ids_state']
                                  for c in updates if c['ids_state'] is not None]
                    if ids_states:
                        ids_avg = {}
                        for key in ids.state_dict().keys():
                            tensors = [torch.tensor(
                                is_[key], dtype=torch.float32) for is_ in ids_states]
                            ids_avg[key] = torch.stack(tensors, 0).mean(0)
                        ids.load_state_dict(ids_avg)

        gen, disc = global_gan_models[input_size]
        ids = global_ids_models.get(input_size)
        package = {
            'new_gen_weights': gen.state_dict(),
            'new_disc_weights': disc.state_dict(),
            'new_ids_weights': ids.state_dict() if ids is not None else None
        }
        send_data(conn, package)
        print(f"[✔] Sent new weights to Client {client_id}")
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


def evaluate_global_model(model, X_test, y_test, batch_size=1024, binary_classification=True):
    """
    Evaluate global model on test data.

    Args:
        model (nn.Module): Model to evaluate.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        batch_size (int): Batch size.
        binary_classification (bool): Binary or multiclass.

    Returns:
        dict: Metrics (accuracy, f1_score).
    """
    from sklearn.metrics import f1_score
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(
            y_test, dtype=torch.float32 if binary_classification else torch.long)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            if binary_classification:
                preds = (outputs.squeeze() > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds,
                  average='macro' if not binary_classification else 'binary')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return {'f1_score': f1, 'accuracy': accuracy}


def run_federated_learning(config):
    """
    Run federated learning server.

    Args:
        config (dict): Server configuration.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((config['host'], config['port']))
    server_socket.listen(config['num_clients'])
    print(
        f"[*] Listening on {config['host']}:{config['port']} for {config['num_clients']} clients")

    client_id = 0
    threads = []
    try:
        while client_id < config['num_clients']:
            conn, addr = server_socket.accept()
            thread = threading.Thread(
                target=handle_client, args=(conn, addr, client_id, config))
            threads.append(thread)
            thread.start()
            client_id += 1

        for thread in threads:
            thread.join()

        print("[*] All clients processed. Evaluating global models...")
        # Example: Evaluate global IDS model (requires test data)
        # for input_size, ids_model in global_ids_models.items():
        #     X_test, y_test = load_test_data(input_size)  # Implement this
        #     metrics = evaluate_global_model(ids_model, X_test, y_test)
        #     print(f"Global IDS (input_size={input_size}): {metrics}")

    except KeyboardInterrupt:
        print("\n[!] Shutting down server...")
    finally:
        server_socket.close()
        print("[*] Server socket closed")


def get_default_config():
    """
    Default server configuration.

    Returns:
        dict: Configuration parameters.
    """
    return {
        'host': '127.0.0.1',
        'port': 8888,
        'num_clients': 1,  # <--- Sửa thành 1 để chỉ chạy 1 client
        'timeout': 60,
        'gan_epochs': 10,
        'ids_epochs': 10,
        'batch_size': 1024,
        'lr_gen': 0.0002,
        'lr_disc': 0.0002,
        'lr_ids': 0.001,
        'penalty_coef': 0.1,
        'num_samples_normal': 100,
        'f1_threshold': 0.9
    }


def main():
    """Main function to start the server."""
    config = get_default_config()
    try:
        run_federated_learning(config)
    except Exception as e:
        print(f"[!] Server failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
