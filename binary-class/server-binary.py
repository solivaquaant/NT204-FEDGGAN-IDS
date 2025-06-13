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


# Global dictionaries to store models and weights
global_gan_models = {}  # input_size: (gen, disc)
global_ids_models = {}  # input_size: ids_model
received_weights = {}   # input_size: [list of updates]
lock = threading.Lock()


def build_global_models(input_size):
    """
    Create global GAN models (Generator and Discriminator).

    Args:
        input_size (int): Feature input size.

    Returns:
        tuple: (GeneratorNet, DiscriminatorNet)
    """
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


def build_global_ids_model(input_size):
    """
    Create global IDS model (CNNClassifier).

    Args:
        input_size (int): Feature input size.

    Returns:
        nn.Module: CNNClassifier
    """
    class CNNClassifier(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Binary classification
            )

        def forward(self, x):
            return self.model(x)
    return CNNClassifier(input_size)


def get_default_config():
    """
    Return default configuration for clients and server.

    Returns:
        dict: Configuration parameters
    """
    return {
        'host': '127.0.0.1',
        'port': 8888,
        'num_clients': 2,  # Number of clients to wait for
        'timeout': 60,     # Timeout in seconds
        'gan_epochs': 10,
        'ids_epochs': 10,
        'batch_size': 1024,
        'num_samples_normal': 10,
        'num_samples_weak': 10,
        'lr_disc': 0.0002,
        'lr_gen': 0.0002,
        'lr_ids': 0.001,
        'penalty_coef': 0.1,
        'labels_to_consider': 15,
        'f1_threshold': 0.9,
        'global_epochs': 5,
        'n_runs': 5
    }


def evaluate_global_model(gen, disc, ids, config, test_data=None):
    """
    Evaluate global GAN and IDS models.

    Args:
        gen (nn.Module): Global Generator
        disc (nn.Module): Global Discriminator
        ids (nn.Module): Global IDS model
        config (dict): Configuration
        test_data (tuple): Optional test dataset (X_test, y_test)

    Returns:
        tuple: (f1_initial, f1_augmented)
    """
    if test_data is None:
        print("[!] No test data provided. Using dummy F1-scores.")
        return 0.5, 0.7  # Dummy values for demonstration

    X_test, y_test = test_data
    # Convert to torch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Evaluate initial IDS
    with torch.no_grad():
        initial_pred = ids(X_test)
        initial_pred = (initial_pred > 0.5).float()
        f1_initial = compute_f1_score(y_test, initial_pred)

    # Generate augmented data
    noise = torch.randn(len(X_test), 5)  # Assuming noise_dim=5
    with torch.no_grad():
        fake_data = gen(noise)
        augmented_pred = ids(fake_data)
        augmented_pred = (augmented_pred > 0.5).float()
        f1_augmented = compute_f1_score(y_test, augmented_pred)

    return f1_initial, f1_augmented


def compute_f1_score(y_true, y_pred):
    """
    Compute F1-score for binary classification.

    Args:
        y_true (torch.Tensor): True labels
        y_pred (torch.Tensor): Predicted labels

    Returns:
        float: F1-score
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true.numpy(), y_pred.numpy())


def run_federated_learning(config):
    """
    Coordinate the Federated Learning process and evaluate performance.

    Args:
        config (dict): Global configuration

    Returns:
        dict: Evaluation results (average improvement, std)
    """
    global received_weights, global_gan_models, global_ids_models
    n_runs = config['n_runs']
    num_clients = config['num_clients']
    num_global_epochs = config['global_epochs']
    avg_improvements = []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")
        # Initialize server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config['host'], config['port']))
        server_socket.listen(num_clients)
        print(f"[*] Server listening on {config['host']}:{config['port']}")

        # Reset global models and weights
        received_weights = defaultdict(list)
        global_gan_models = {}
        global_ids_models = {}
        client_f1_scores = []

        for epoch in range(num_global_epochs):
            print(f"\n--- Global Epoch {epoch+1}/{num_global_epochs} ---")
            # Accept connections from clients
            threads = []
            for client_id in range(num_clients):
                conn, addr = server_socket.accept()
                thread = threading.Thread(
                    target=handle_client, args=(conn, addr, client_id, config))
                threads.append(thread)
                thread.start()

            # Wait for all clients to complete
            for thread in threads:
                thread.join()

            # Evaluate global model after last epoch
            if epoch == num_global_epochs - 1:
                for input_size, (gen, disc) in global_gan_models.items():
                    ids = global_ids_models.get(input_size)
                    f1_initial, f1_augmented = evaluate_global_model(
                        gen, disc, ids, config)
                    improvement = np.mean(f1_augmented) - np.mean(f1_initial)
                    client_f1_scores.append(improvement)

        avg_improvements.append(np.mean(client_f1_scores))
        print(f"Run {run+1} Improvement: {np.mean(client_f1_scores):.4f}")

        # Close server socket
        server_socket.close()

    # Summarize results
    print(
        f"\n>>> Average Improvement over {n_runs} runs: {np.mean(avg_improvements):.4f} ± {np.std(avg_improvements):.4f}")
    return {
        'avg_improvement': np.mean(avg_improvements),
        'std_improvement': np.std(avg_improvements)
    }


def handle_client(conn, addr, client_id, config):
    """Handle a client connection."""
    print(f"[+] Client {client_id} connected from {addr}")
    try:
        # 1. Send configuration to client
        config_dict = copy.deepcopy(config)
        send_data(conn, config_dict)
        print(f"[=] Sent config to Client {client_id}")

        # 2. Receive parameters from client (GAN and IDS)
        data = receive_data(conn)
        input_size = data.get('input_size')
        gen_state = data.get('gen_state')
        disc_state = data.get('disc_state')
        ids_state = data.get('ids_state', None)  # Optional IDS state

        if input_size is None or gen_state is None or disc_state is None:
            raise ValueError(f"Invalid data received from Client {client_id}")

        print(
            f"[=] Received local weights from Client {client_id} (input_size={input_size})")
        with lock:
            if input_size not in global_gan_models:
                global_gan_models[input_size] = build_global_models(input_size)
            if input_size not in global_ids_models and ids_state is not None:
                global_ids_models[input_size] = build_global_ids_model(
                    input_size)

            received_weights[input_size].append({
                'gen_state': gen_state,
                'disc_state': disc_state,
                'ids_state': ids_state
            })

        # 3. Wait for other clients
        wait_start = time.time()
        while sum(len(lst) for lst in received_weights.values()) < config['num_clients']:
            time.sleep(0.1)
            if time.time() - wait_start > config['timeout']:
                print(
                    f"[!] Timeout waiting for clients. Proceeding with {sum(len(lst) for lst in received_weights.values())} clients.")
                break

        # 4. Federated Averaging
        with lock:
            for isize, updates in received_weights.items():
                gen, disc = global_gan_models[isize]
                ids = global_ids_models.get(isize)

                # Aggregate Generator
                gen_states = [c['gen_state'] for c in updates]
                gen_avg = {}
                for key in gen.state_dict().keys():
                    tensors = [torch.tensor(gs[key], dtype=torch.float32)
                               for gs in gen_states]
                    gen_avg[key] = torch.stack(tensors, 0).mean(0)
                gen.load_state_dict(gen_avg)

                # Aggregate Discriminator
                disc_states = [c['disc_state'] for c in updates]
                disc_avg = {}
                for key in disc.state_dict().keys():
                    tensors = [torch.tensor(ds[key], dtype=torch.float32)
                               for ds in disc_states]
                    disc_avg[key] = torch.stack(tensors, 0).mean(0)
                disc.load_state_dict(disc_avg)

                # Aggregate IDS (if available)
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

        # 5. Send global parameters back to client
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


def send_data(conn, data):
    """Send serialized data over the connection."""
    try:
        serialized = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(serialized)))
        conn.sendall(serialized)
        print(f"Sent {len(serialized)} bytes")
        return True
    except Exception as e:
        print(f"Error sending data: {e}")
        return False


def receive_data(conn):
    """Receive serialized data from the connection."""
    try:
        raw_length = conn.recv(4)
        if not raw_length:
            raise ConnectionError("Client closed connection (length)")
        length = struct.unpack('!I', raw_length)[0]
        print(f"Expecting to receive {length} bytes")

        data = b""
        while len(data) < length:
            packet = conn.recv(min(4096, length - len(data)))
            if not packet:
                raise ConnectionError(
                    f"Incomplete data received: {len(data)}/{length} bytes")
            data += packet
            print(f"Received {len(data)}/{length} bytes")

        return pickle.loads(data)
    except Exception as e:
        print(f"Error receiving data: {e}")
        traceback.print_exc()
        raise


def main():
    """Main function to start the server."""
    config = get_default_config()
    try:
        run_federated_learning(config)
    except KeyboardInterrupt:
        print("\nServer terminated by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
