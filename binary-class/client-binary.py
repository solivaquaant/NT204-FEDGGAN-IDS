import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import traceback
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import gzip
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Example configuration (replace with your GANIDSConfig)


class GANIDSConfig:
    def __init__(self):
        self.dataset_type = 'NSL-KDD'
        self.gan_epochs = 10
        self.ids_epochs = 10
        self.batch_size = 1024
        self.lr_gen = 0.0002
        self.lr_disc = 0.0002
        self.lr_ids = 0.001
        self.penalty_coef = 0.1
        self.num_samples_normal = 100
        self.f1_threshold = 0.9
        self.client_id = 0
        self.binary_classification = True
        self.non_iid = True

    def get_dataset_path(self):
        # return r'.\nsl-kdd-data\NSL_KDD_Train.csv'
        return r'.\unsw-nb15\UNSW_NB15_training-set.csv'

    def get_test_dataset_path(self):
        # return r'.\nsl-kdd-data\NSL_KDD_Test.csv'
        return r'.\unsw-nb15\UNSW_NB15_testing-set.csv'

    def get_target_column(self):
        return 'label'

# Assuming GANIDSConfig is provided by the user
# Example: class GANIDSConfig:
#     gan_epochs, ids_epochs, batch_size, lr_gen, lr_disc, lr_ids, penalty_coef, ...


class Logger:
    """Simple logger to print and save logs."""

    def __init__(self, log_file='client.log'):
        self.log_file = log_file

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{time.ctime()}: {message}\n")


logger = Logger()


def create_labels(label_type, size=None, labels=None, dtype=torch.float32):
    """
    Create label tensors for GAN or IDS.

    Args:
        label_type (str): Type of labels ("ones", "zeros", "class").
        size (int): Number of labels (for "ones" or "zeros").
        labels (array-like): Input labels (for "class").
        dtype (torch.dtype): Tensor data type.

    Returns:
        torch.Tensor: Label tensor.
    """
    if label_type == "ones":
        if size is None:
            raise ValueError("Size must be provided for label_type='ones'")
        return torch.ones(size, 1, dtype=dtype)
    elif label_type == "zeros":
        if size is None:
            raise ValueError("Size must be provided for label_type='zeros'")
        return torch.zeros(size, 1, dtype=dtype)
    elif label_type == "class":
        if labels is None:
            raise ValueError("Labels must be provided for label_type='class'")
        return torch.tensor(labels, dtype=dtype).view(-1, 1)
    else:
        raise ValueError(
            "Invalid label_type. Must be 'ones', 'zeros', or 'class'.")


def noise(size, noise_dim=5, dtype=torch.float32):
    """
    Generate random noise for Generator.

    Args:
        size (int): Number of noise samples.
        noise_dim (int): Noise dimension (default: 5).
        dtype (torch.dtype): Tensor data type.

    Returns:
        torch.Tensor: Noise tensor of shape (size, noise_dim).
    """
    return torch.randn(size, noise_dim, dtype=dtype)


class Data_Loader(Dataset):
    """Custom Dataset for loading data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(X, y, batch_size):
    """
    Create DataLoader from features and labels.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        batch_size (int): Batch size.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = Data_Loader(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class GeneratorNet(nn.Module):
    """Generator network for GAN."""

    def __init__(self, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 50),  # noise_dim=5
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module):
    """Discriminator network for GAN."""

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


class CNNClassifier(nn.Module):
    """IDS classifier network."""

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


def load_and_preprocess_data(config: GANIDSConfig):
    """
    Load and preprocess data for GAN and IDS training.

    Args:
        config (GANIDSConfig): Configuration with dataset parameters.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, input_size)
            - X_train (np.ndarray): Training features, shape (n_samples, n_features).
            - y_train (np.ndarray): Training labels, shape (n_samples,).
            - X_test (np.ndarray): Test features, shape (m_samples, n_features).
            - y_test (np.ndarray): Test labels, shape (m_samples,).
            - input_size (int): Number of features (n_features).
    """
    try:
        # Get parameters from config
        path = config.get_dataset_path()
        test_path = config.get_test_dataset_path()
        target_column = config.get_target_column()
        dataset_type = config.dataset_type
        client_id = getattr(config, 'client_id', 0)  # For Non-IID filtering
        binary_classification = getattr(
            config, 'binary_classification', True)  # Binary or multiclass

        # Check file existence
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test dataset file not found: {test_path}")

        # Define features for each dataset
        if dataset_type in ['KDD99', 'NSL-KDD']:
            features = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
            ]
            categorical_columns = ['protocol_type', 'service', 'flag']
            input_size = 42
        elif dataset_type == 'UNSW-NB15':
            # Assuming standard UNSW-NB15 features (adjust based on your CSV)
            features = [
                'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
                'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
                'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
                'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd',
                'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat'
            ]
            categorical_columns = ['proto', 'service', 'state']
            input_size = 45
        else:
            raise ValueError(f"Unsupported dataset: {dataset_type}")

        # Load data
        if dataset_type in ['KDD99', 'NSL-KDD']:
            if path.endswith('.gz'):
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, names=features, header=None)
                with gzip.open(test_path, 'rt') as f:
                    df_test = pd.read_csv(f, names=features, header=None)
            else:
                df = pd.read_csv(path, names=features, header=None)
                df_test = pd.read_csv(test_path, names=features, header=None)
        else:  # UNSW-NB15
            df = pd.read_csv(path)
            df_test = pd.read_csv(test_path)

        # Clean data
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df_test = df_test.dropna().drop_duplicates().reset_index(drop=True)

        # Validate target column
        if target_column not in df.columns or target_column not in df_test.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset")

        # Encode categorical columns
        label_encoders = {}
        for col in categorical_columns:
            if col in df.columns and col in df_test.columns:
                le = LabelEncoder()
                unique_values = pd.concat(
                    [df[col].astype(str), df_test[col].astype(str)]).unique()
                le.fit(unique_values)
                df[col] = le.transform(df[col].astype(str))
                df_test[col] = le.transform(df_test[col].astype(str))
                label_encoders[col] = le

        # Normalize numerical features
        numerical_features = [
            col for col in df.columns if col not in categorical_columns + [target_column]]

        def check_numeric(df, features):
            non_numeric = [
                col for col in features if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"Non-numeric columns: {non_numeric}")

        check_numeric(df, numerical_features)
        check_numeric(df_test, numerical_features)

        scaler = MinMaxScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        df_test[numerical_features] = scaler.transform(
            df_test[numerical_features])

        # Encode labels
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(df[target_column])
        test_labels = df_test[target_column]

        valid_mask = test_labels.isin(label_encoder.classes_)
        if not valid_mask.all():
            unseen_labels = test_labels[~valid_mask].unique()
            print(
                f"Warning: Filtering out {sum(~valid_mask)} test samples with unseen labels: {unseen_labels}")
            df_test = df_test[valid_mask].reset_index(drop=True)
            y_test = label_encoder.transform(df_test[target_column])
        else:
            y_test = label_encoder.transform(test_labels)

        # Convert to binary labels if required
        if binary_classification and dataset_type in ['KDD99', 'NSL-KDD']:
            # Example: normal=0, all attacks=1
            normal_label = label_encoder.transform(['normal'])[0]
            y_train = (y_train != normal_label).astype(int)
            y_test = (y_test != normal_label).astype(int)
        elif binary_classification and dataset_type == 'UNSW-NB15':
            # Adjust based on attack_cat or label column
            normal_label = label_encoder.transform(['Normal'])[0]
            y_train = (y_train != normal_label).astype(int)
            y_test = (y_test != normal_label).astype(int)

        # Extract features
        X_train = df.drop(columns=[target_column]).to_numpy()
        X_test = df_test.drop(columns=[target_column]).to_numpy()

        input_size = X_train.shape[1]  # Tính động số đặc trưng

        # Log để debug
        print(f"Calculated input_size: {X_train.shape[1]}")
        print(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")

        # Simulate Non-IID (example: each client gets a subset of classes)
        if hasattr(config, 'non_iid') and config.non_iid:
            np.random.seed(client_id)
            classes = np.unique(y_train)
            num_classes = min(2, len(classes))  # Example: 2 classes per client
            selected_classes = np.random.choice(
                classes, num_classes, replace=False)
            mask = np.isin(y_train, selected_classes)
            X_train = X_train[mask]
            y_train = y_train[mask]
            print(f"Client {client_id} Non-IID classes: {selected_classes}")

        # Log data info
        print(
            f"Training data shape: {X_train.shape}, Classes: {np.unique(y_train)}")
        print(
            f"Testing data shape: {X_test.shape}, Classes: {np.unique(y_test)}")
        print(
            f"Class distribution (train):\n{pd.Series(y_train).value_counts()}")
        print(
            f"Class distribution (test):\n{pd.Series(y_test).value_counts()}")

        return X_train, y_train, X_test, y_test, input_size

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def train_discriminator(discriminator, optimizer, real_data, fake_data, penalty_coef):
    """
    Train Discriminator for one step.

    Args:
        discriminator (nn.Module): Discriminator model.
        optimizer (torch.optim): Optimizer.
        real_data (torch.Tensor): Real data batch.
        fake_data (torch.Tensor): Fake data batch.
        penalty_coef (float): Penalty coefficient for gradient penalty.

    Returns:
        float: Discriminator loss.
    """
    optimizer.zero_grad()

    # Real data
    real_pred = discriminator(real_data)
    real_labels = create_labels("ones", size=real_data.size(0))
    real_loss = nn.BCELoss()(real_pred, real_labels)

    # Fake data
    fake_pred = discriminator(fake_data.detach())
    fake_labels = create_labels("zeros", size=fake_data.size(0))
    fake_loss = nn.BCELoss()(fake_pred, fake_labels)

    # Gradient penalty (simplified)
    alpha = torch.rand(real_data.size(0), 1)
    interpolates = alpha * real_data + (1 - alpha) * fake_data.detach()
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient_penalty = penalty_coef * \
        ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    # Total loss
    loss = real_loss + fake_loss + gradient_penalty
    loss.backward()
    optimizer.step()

    return loss.item()


def train_generator(generator, discriminator, optimizer, batch_size):
    """
    Train Generator for one step.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        optimizer (torch.optim): Optimizer.
        batch_size (int): Size of input noise batch.

    Returns:
        float: Generator loss.
    """
    optimizer.zero_grad()

    noise_input = noise(batch_size)
    fake_data = generator(noise_input)
    fake_pred = discriminator(fake_data)
    real_labels = create_labels("ones", size=batch_size)

    loss = nn.BCELoss()(fake_pred, real_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_gan_local(generator, discriminator, dataloader, config):
    """
    Train GAN locally on client.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        dataloader (DataLoader): DataLoader for real data.
        config (GANIDSConfig): Configuration.

    Returns:
        tuple: (generator, discriminator) updated models.
    """
    optimizer_gen = optim.Adam(generator.parameters(), lr=config.lr_gen)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=config.lr_disc)

    for epoch in range(config.gan_epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)

            # Train Discriminator
            noise_input = noise(batch_size)
            fake_data = generator(noise_input)
            disc_loss = train_discriminator(
                discriminator, optimizer_disc, real_data, fake_data, config.penalty_coef
            )

            # Train Generator
            gen_loss = train_generator(
                generator, discriminator, optimizer_gen, batch_size)

        logger.log(
            f"Epoch {epoch+1}/{config.gan_epochs}: D_Loss={disc_loss:.4f}, G_Loss={gen_loss:.4f}")

    return generator, discriminator


def train_ids_model(ids_model, dataloader, config, augmented_data=None):
    """
    Train IDS model locally.

    Args:
        ids_model (nn.Module): IDS model.
        dataloader (DataLoader): DataLoader for real data.
        config (GANIDSConfig): Configuration.
        augmented_data (tuple): Optional (X_aug, y_aug) from Generator.

    Returns:
        tuple: (ids_model, history) where history contains F1-scores.
    """
    optimizer = optim.Adam(ids_model.parameters(), lr=config.lr_ids)
    criterion = nn.BCELoss()
    history = {'f1_scores': []}

    for epoch in range(config.ids_epochs):
        ids_model.train()
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = ids_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on test data (assuming dataloader has test data)
        ids_model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for data, labels in dataloader:
                preds = ids_model(data)
                preds = (preds > 0.5).float()
                all_preds.extend(preds.numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
            f1 = f1_score(all_labels, all_preds)
            history['f1_scores'].append(f1)

        logger.log(f"IDS Epoch {epoch+1}/{config.ids_epochs}: F1={f1:.4f}")

    # Train with augmented data if provided
    if augmented_data is not None:
        X_aug, y_aug = augmented_data
        aug_dataloader = create_dataloader(X_aug, y_aug, config.batch_size)
        for epoch in range(config.ids_epochs):
            ids_model.train()
            for data, labels in aug_dataloader:
                optimizer.zero_grad()
                outputs = ids_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate
            ids_model.eval()
            with torch.no_grad():
                all_preds, all_labels = [], []
                for data, labels in aug_dataloader:
                    preds = ids_model(data)
                    preds = (preds > 0.5).float()
                    all_preds.extend(preds.numpy().flatten())
                    all_labels.extend(labels.numpy().flatten())
                f1 = f1_score(all_labels, all_preds)
                history['f1_scores'].append(f1)

            logger.log(
                f"Augmented IDS Epoch {epoch+1}/{config.ids_epochs}: F1={f1:.4f}")

    return ids_model, history


def evaluate_local_model(model, dataloader):
    """
    Evaluate model on local test data.

    Args:
        model (nn.Module): Model to evaluate (Discriminator or IDS).
        dataloader (DataLoader): DataLoader for test data.

    Returns:
        dict: Metrics (accuracy, f1_score).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            preds = model(data)
            preds = (preds > 0.5).float()
            all_preds.extend(preds.numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    f1 = f1_score(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return {'f1_score': f1, 'accuracy': accuracy}


def plot_f1_scores(initial_history, augmented_history):
    """
    Plot F1-scores for initial and augmented IDS.

    Args:
        initial_history (dict): History of initial IDS.
        augmented_history (dict): History of augmented IDS.
    """
    plt.plot(initial_history['f1_scores'], label='Initial IDS')
    plt.plot(augmented_history['f1_scores'], label='Augmented IDS')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Comparison')
    plt.legend()
    plt.savefig('f1_scores.png')
    plt.close()


def plot_bar_comparison(initial_metrics, augmented_metrics, client_id, save_path='bar_comparison.png'):
    """
    Plot bar chart comparing F1-score and accuracy between initial and augmented IDS.

    Args:
        initial_metrics (dict): Metrics of initial IDS {'f1_score', 'accuracy'}
        augmented_metrics (dict): Metrics of augmented IDS
        client_id (int): Client identifier
        save_path (str): Path to save the plot
    """
    import matplotlib.pyplot as plt

    labels = ['F1-Score', 'Accuracy']
    initial_values = [initial_metrics['f1_score'], initial_metrics['accuracy']]
    augmented_values = [augmented_metrics['f1_score'],
                        augmented_metrics['accuracy']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, initial_values, width,
           label='Initial IDS', color='#1f77b4')
    ax.bar(x + width/2, augmented_values, width,
           label='Augmented IDS', color='#ff7f0e')

    ax.set_ylabel('Score')
    ax.set_title(f'Performance Comparison for Client {client_id}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.log(f"Saved bar comparison plot to {save_path}")


def plot_line_f1_scores(initial_history, augmented_history, client_id, save_path='line_f1_scores.png'):
    """
    Plot line chart of F1-scores over epochs for initial and augmented IDS.

    Args:
        initial_history (dict): History of initial IDS {'f1_scores'}
        augmented_history (dict): History of augmented IDS
        client_id (int): Client identifier
        save_path (str): Path to save the plot
    """
    import matplotlib.pyplot as plt

    plt.plot(initial_history['f1_scores'],
             label='Initial IDS', color='#1f77b4', marker='o')
    plt.plot(augmented_history['f1_scores'],
             label='Augmented IDS', color='#ff7f0e', marker='s')

    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title(f'F1-Score over Epochs for Client {client_id}')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    logger.log(f"Saved line F1-scores plot to {save_path}")


def connect_to_server(host, port):
    """
    Connect to the server.

    Args:
        host (str): Server host.
        port (int): Server port.

    Returns:
        socket.socket: Connected socket.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    logger.log(f"Connected to server at {host}:{port}")
    return client_socket


def send_data(conn, data):
    """
    Send serialized data to server.

    Args:
        conn (socket.socket): Connection socket.
        data: Data to send.

    Returns:
        bool: Success status.
    """
    try:
        serialized = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(serialized)))
        conn.sendall(serialized)
        logger.log(f"Sent {len(serialized)} bytes to server")
        return True
    except Exception as e:
        logger.log(f"Error sending data: {e}")
        return False


def receive_data(conn):
    """
    Receive serialized data from server.

    Args:
        conn (socket.socket): Connection socket.

    Returns:
        Data received.
    """
    try:
        raw_length = conn.recv(4)
        if not raw_length:
            raise ConnectionError("Server closed connection")
        length = struct.unpack('!I', raw_length)[0]
        logger.log(f"Expecting to receive {length} bytes")

        data = b""
        while len(data) < length:
            packet = conn.recv(min(4096, length - len(data)))
            if not packet:
                raise ConnectionError(
                    f"Incomplete data: {len(data)}/{length} bytes")
            data += packet
            logger.log(f"Received {len(data)}/{length} bytes")

        return pickle.loads(data)
    except Exception as e:
        logger.log(f"Error receiving data: {e}")
        raise


def update_local_model(generator, discriminator, ids_model, global_weights):
    """
    Update local models with global weights.

    Args:
        generator (nn.Module): Local Generator.
        discriminator (nn.Module): Local Discriminator.
        ids_model (nn.Module): Local IDS model.
        global_weights (dict): Global weights from server.

    Returns:
        tuple: Updated (generator, discriminator, ids_model).
    """
    if global_weights['new_gen_weights']:
        generator.load_state_dict(global_weights['new_gen_weights'])
    if global_weights['new_disc_weights']:
        discriminator.load_state_dict(global_weights['new_disc_weights'])
    if global_weights['new_ids_weights']:
        ids_model.load_state_dict(global_weights['new_ids_weights'])
    logger.log("Updated local models with global weights")
    return generator, discriminator, ids_model


def run_gan_ids_pipeline(config, client_id, host, port, verbose=True):
    """
    Run the GAN-IDS pipeline on the client.

    Args:
        config (GANIDSConfig): Configuration.
        client_id (int): Client identifier.
        host (str): Server host.
        port (int): Server port.
        verbose (bool): Print verbose output.

    Returns:
        dict: Results including model states and metrics.
    """
    logger.log(f"Starting GAN-IDS pipeline for Client {client_id}")

    # Connect to server
    conn = connect_to_server(host, port)

    try:
        # Receive configuration from server
        server_config = receive_data(conn)
        # Update local config with server config
        for key, value in server_config.items():
            setattr(config, key, value) if hasattr(config, key) else None
        logger.log("Received and applied server configuration")

        X_train, y_train, X_test, y_test, input_size = load_and_preprocess_data(
            config)

        # Create DataLoaders
        train_dataloader = create_dataloader(
            X_train, y_train, config.batch_size)
        test_dataloader = create_dataloader(X_test, y_test, config.batch_size)

        # Log để debug
        logger.log(
            f"Client {client_id} loaded data with input_size={input_size}")

        # Initialize models
        generator = GeneratorNet(input_size)
        discriminator = DiscriminatorNet(input_size)
        ids_model = CNNClassifier(input_size)

        # Train GAN locally
        generator, discriminator = train_gan_local(
            generator, discriminator, train_dataloader, config)

        # Generate augmented data
        noise_input = noise(config.num_samples_normal)
        with torch.no_grad():
            X_aug = generator(noise_input).numpy()
            y_aug = np.ones(config.num_samples_normal)  # Assume normal class
        aug_dataloader = create_dataloader(X_aug, y_aug, config.batch_size)

        # Train initial IDS
        initial_ids, initial_history = train_ids_model(
            ids_model, train_dataloader, config)

        # Train augmented IDS
        augmented_ids, augmented_history = train_ids_model(
            ids_model, train_dataloader, config, augmented_data=(X_aug, y_aug)
        )

        # Evaluate models
        disc_metrics = evaluate_local_model(discriminator, test_dataloader)
        initial_metrics = evaluate_local_model(initial_ids, test_dataloader)
        augmented_metrics = evaluate_local_model(
            augmented_ids, test_dataloader)

        # Plot F1-scores
        if verbose:
            plot_f1_scores(initial_history, augmented_history)

        # Prepare data to send to server
        package = {
            'input_size': input_size,
            'gen_state': generator.state_dict(),
            'disc_state': discriminator.state_dict(),
            'ids_state': augmented_ids.state_dict(),
            'metrics': {
                'discriminator': disc_metrics,
                'initial_ids': initial_metrics,
                'augmented_ids': augmented_metrics
            }
        }
        send_data(conn, package)
        logger.log(
            f"Sent local weights and metrics to server for Client {client_id}")

        # Receive global weights
        global_weights = receive_data(conn)
        generator, discriminator, ids_model = update_local_model(
            generator, discriminator, ids_model, global_weights
        )

        # Prepare results
        results = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'ids_model': ids_model.state_dict(),
            'initial_ids': initial_history,
            'augmented_ids': augmented_history,
            'metrics': package['metrics']
        }

        return results

    except Exception as e:
        logger.log(f"Error in pipeline for Client {client_id}: {e}")
        traceback.print_exc()
        raise
    finally:
        conn.close()
        logger.log(f"Connection closed for Client {client_id}")


def main():
    """Main function to start the client."""
    config = GANIDSConfig()
    client_id = 0  # Should be unique for each client
    host = '127.0.0.1'
    port = 8888

    try:
        results = run_gan_ids_pipeline(
            config, client_id, host, port, verbose=True)
        logger.log(f"Client {client_id} completed pipeline successfully")
    except Exception as e:
        logger.log(f"Client {client_id} failed: {e}")


if __name__ == '__main__':
    main()
