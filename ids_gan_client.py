import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, classification_report
import re
import warnings
import gzip
import os
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import sys


class GANIDSConfig:
    def __init__(self):
        self.gen_params = None
        self.disc_params = None
        self.local_epochs = 10
        self.global_epochs = 50
        self.batch_size = 1000
        self.num_samples_normal = 10
        self.num_samples_weak = 50
        self.lr_disc = 0.0002
        self.lr_gen = 0.0002
        self.penalty_coef = 0.1
        self.pca_components = 20
        self.labels_to_consider = 15
        self.dataset_type = 'UNSW-NB15'
        self.categorical_columns = []
        # Map dataset types to their target columns
        self.target_column_map = {
            'KDD99': 'label',
            'NSL-KDD': 'label',
            'UNSW-NB15': 'attack_cat'
        }
        self.feature_columns = []
        self.f1_threshold = 0.9
        self.dataset_paths = {
            'KDD99': 'kdd-cup-1999-data/kddcup.data_10_percent.gz',
            'NSL-KDD': 'nsl-kdd-data/NSL_KDD_Train.csv',
            'UNSW-NB15': 'unsw-nb15/UNSW_NB15_training-set.csv'
        }
        self.test_dataset_paths = {
            'KDD99': 'kdd-cup-1999-data/kddcup.data_10_percent_corrected',
            'NSL-KDD': 'nsl-kdd-data/NSL_KDD_Test.csv',
            'UNSW-NB15': 'unsw-nb15/UNSW_NB15_testing-set.csv'
        }

    def set_training_params(self, params_dict: Dict[str, Any]) -> None:
        valid_params = [
            'local_epochs', 'global_epochs', 'batch_size', 'num_samples_normal', 'num_samples_weak',
            'lr_disc', 'lr_gen', 'penalty_coef', 'pca_components',
            'labels_to_consider', 'f1_threshold', 'dataset_type'
        ]
        for param, value in params_dict.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                warnings.warn(
                    f"Parameter '{param}' is not recognized and will be ignored.")

    def get_dataset_path(self) -> str:
        return self.dataset_paths[self.dataset_type]

    def get_test_dataset_path(self) -> str:
        return self.test_dataset_paths[self.dataset_type]

    def get_target_column(self) -> str:
        """Return the target column name for the current dataset type."""
        if self.dataset_type not in self.target_column_map:
            raise ValueError(
                f"No target column defined for dataset type: {self.dataset_type}")
        return self.target_column_map[self.dataset_type]

    def prompt_dataset_type(self):
        print("Chọn loại dataset:")
        for i, name in enumerate(self.dataset_paths.keys(), 1):
            print(f"{i}. {name}")

        while True:
            try:
                choice = int(input("Nhập số tương ứng (ví dụ: 1): "))
                dataset_names = list(self.dataset_paths.keys())
                if 1 <= choice <= len(dataset_names):
                    self.dataset_type = dataset_names[choice - 1]
                    print(f">>> Đã chọn dataset: {self.dataset_type}")
                    break
                else:
                    print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
            except ValueError:
                print("Nhập số nguyên hợp lệ.")


def validate_config(config):
    """Validate configuration parameters"""
    errors = []

    if config.pca_components <= 0:
        errors.append("PCA components must be positive")

    if config.labels_to_consider <= 0:
        errors.append("Labels to consider must be positive")

    if not 0 < config.f1_threshold <= 1:
        errors.append("F1 threshold must be between 0 and 1")

    if config.batch_size <= 0:
        errors.append("Batch size must be positive")

    if config.local_epochs <= 0 or config.global_epochs <= 0:
        errors.append("Epochs must be positive")

    if config.dataset_type not in config.dataset_paths:
        errors.append(f"Unsupported dataset type: {config.dataset_type}")

    if errors:
        raise ValueError("Configuration errors:\n" +
                         "\n".join(f"- {error}" for error in errors))

    return True


def load_and_preprocess_data(config: GANIDSConfig):
    try:
        path = config.get_dataset_path()
        test_path = config.get_test_dataset_path()
        config.target_column = config.get_target_column()

        # Check if files exist
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test dataset file not found: {test_path}")

        # Dataset-specific loading logic
        if config.dataset_type == 'KDD99':
            features_kdd = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
            ]
            if path.endswith('.gz'):
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, names=features_kdd, header=None)
            else:
                df = pd.read_csv(path, names=features_kdd, header=None)
            config.categorical_columns = ['protocol_type', 'service', 'flag']

            # Load test dataset
            if test_path.endswith('.gz'):
                with gzip.open(test_path, 'rt') as f:
                    df_test = pd.read_csv(f, names=features_kdd, header=None)
            else:
                df_test = pd.read_csv(
                    test_path, names=features_kdd, header=None)

        elif config.dataset_type == 'NSL-KDD':
            features_kdd = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'drop'
            ]
            df = pd.read_csv(path, names=features_kdd, header=None)
            df_test = pd.read_csv(test_path, names=features_kdd, header=None)
            config.categorical_columns = ['protocol_type', 'service', 'flag']

        elif config.dataset_type == 'UNSW-NB15':
            df = pd.read_csv(path)
            df_test = pd.read_csv(test_path)
            config.categorical_columns = [col for col in [
                'proto', 'service', 'state'] if col in df.columns]

        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_type}")

        # Validate target column
        if config.target_column not in df.columns or config.target_column not in df_test.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found in dataset {config.dataset_type}")

        # Clean data
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df_test = df_test.dropna().drop_duplicates().reset_index(drop=True)

        # Encode categorical columns
        label_encoders = {}
        for col in config.categorical_columns:
            if col in df.columns and col in df_test.columns:
                le = LabelEncoder()
                # Fit on combined unique values from train and test
                unique_values = pd.concat(
                    [df[col].astype(str), df_test[col].astype(str)]).unique()
                le.fit(unique_values)
                df[col] = le.transform(df[col].astype(str))
                df_test[col] = le.transform(df_test[col].astype(str))
                label_encoders[col] = le

        # Normalization
        scaler = MinMaxScaler()

        def check_numeric(df, features, dataset_name):
            non_numeric = [
                col for col in features if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(
                    f"{dataset_name}: The following columns are not numeric types: {non_numeric}")

        if config.dataset_type in ['KDD99', 'NSL-KDD']:
            numerical_features_kdd = [
                'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate'
            ]
            valid_features = [
                col for col in numerical_features_kdd if col in df.columns]
            check_numeric(df, valid_features, "KDD99/NSL-KDD")
            check_numeric(df_test, valid_features, "KDD99/NSL-KDD")
            # Fit scaler on training data only
            scaler.fit(df[valid_features])
            df[valid_features] = scaler.transform(df[valid_features])
            df_test[valid_features] = scaler.transform(df_test[valid_features])
        else:
            numerical_features_unsw = [
                'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
                'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
                'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                'ct_srv_dst'
            ]
            valid_features = [
                col for col in numerical_features_unsw if col in df.columns]
            check_numeric(df, valid_features, "UNSW-NB15")
            check_numeric(df_test, valid_features, "UNSW-NB15")
            scaler.fit(df[valid_features])
            df[valid_features] = scaler.transform(df[valid_features])
            df_test[valid_features] = scaler.transform(df_test[valid_features])

        # Define reshape dimensions
        if config.dataset_type in ['KDD99', 'NSL-KDD']:
            shape = (7, 6)
        else:
            shape = (9, 5)

        # Extract and reshape features
        def reshape_features(X, shape):
            padded = np.zeros((X.shape[0], shape[0] * shape[1]))
            padded[:, :min(X.shape[1], shape[0] * shape[1])] = X[:,
                                                                 :min(X.shape[1], shape[0] * shape[1])]
            return padded.reshape((-1, shape[0], shape[1], 1))

        # Encode labels
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(df[config.target_column])
        # Filter test data to only include labels seen in training
        test_labels = df_test[config.target_column]
        valid_mask = test_labels.isin(label_encoder.classes_)
        if not valid_mask.all():
            unseen_labels = test_labels[~valid_mask].unique()
            print(
                f"Warning: Filtering out {sum(~valid_mask)} test samples with unseen labels: {unseen_labels}")
            df_test = df_test[valid_mask].reset_index(drop=True)

        y_test = label_encoder.transform(df_test[config.target_column])

        # Extract features after filtering
        X_train = df.drop(columns=[config.target_column]).to_numpy()
        X_test = df_test.drop(columns=[config.target_column]).to_numpy()
        x_train_reshaped = reshape_features(X_train, shape)
        x_test_reshaped = reshape_features(X_test, shape)

        # Create DataFrames with reshaped features
        feature_columns = [f'feature_{i}' for i in range(
            x_train_reshaped.shape[1] * x_train_reshaped.shape[2])]
        train_df = pd.DataFrame(x_train_reshaped.reshape(
            x_train_reshaped.shape[0], -1), columns=feature_columns)
        test_df = pd.DataFrame(x_test_reshaped.reshape(
            x_test_reshaped.shape[0], -1), columns=feature_columns)
        train_df[config.target_column] = y_train
        test_df[config.target_column] = y_test

        print(
            f"Training dataset shape: {x_train_reshaped.shape}, Classes: {np.unique(y_train)}")
        print(
            f"Testing dataset shape: {x_test_reshaped.shape}, Classes: {np.unique(y_test)}")
        print(
            f"Class distribution (train):\n{pd.Series(y_train).value_counts()}")
        print(
            f"Class distribution (test):\n{pd.Series(y_test).value_counts()}")

        return x_train_reshaped, x_test_reshaped, y_train, y_test, train_df, test_df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


class Data_Loader(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]

        if isinstance(img, torch.Tensor):
            img_tensor = img.detach().clone().float()
        else:
            img_tensor = torch.tensor(img, dtype=torch.float32)

        label = self.data[index][1]
        return (img_tensor, label)


def noise(size):
    return torch.randn(size, 5)


def ones_target(size):
    return torch.ones(size, 1)


def zeros_target(size):
    return torch.zeros(size, 1)


def true_target(y):
    return torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 2. Fix data loader issue - đảm bảo consistent data types


def create_dataloader(x, y, batch_size, shuffle=True):
    # Handle 4D input by flattening to 2D
    if x.ndim == 4:
        x = x.reshape(x.shape[0], -1)  # Flatten to (n_samples, 42)
        print(f"Flattened input shape: {x.shape}")

    # Ensure tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    dataset = Data_Loader(list(zip(x, y)))
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    print(
        f"Created DataLoader with batch_size={batch_size}, shuffle={shuffle}")
    return dataloader


def prepare_datasets(dataset: pd.DataFrame, config: GANIDSConfig):
    """
    Tách dữ liệu theo số lớp tối đa cho phép, train/test split từng lớp.
    Trả về dữ liệu huấn luyện, kiểm thử, và bản sao của DataFrame train/test.
    """
    class_dfs = []
    labels = dataset[config.target_column].unique()

    for label in labels:
        df_label = dataset[dataset[config.target_column] == label]
        if len(df_label) < 10:
            print(f"Bỏ qua lớp {label}: quá ít mẫu ({len(df_label)})")
            continue
        class_dfs.append((label, df_label))

    # Sort theo số lượng mẫu và chọn top N class
    class_dfs.sort(key=lambda x: len(x[1]), reverse=True)
    selected_classes = class_dfs[:config.labels_to_consider]

    # Chỉ giữ lại các mẫu thuộc các lớp đã chọn
    allowed_labels = [label for label, _ in selected_classes]
    dataset = dataset[dataset[config.target_column].isin(allowed_labels)]

    train_frames, test_frames = [], []

    for label in allowed_labels:
        df_label = dataset[dataset[config.target_column] == label]
        train_df, test_df = train_test_split(
            df_label,
            test_size=0.25,
            random_state=42,
            stratify=df_label[[config.target_column]]
        )
        train_frames.append(train_df)
        test_frames.append(test_df)

    train_df = pd.concat(train_frames).reset_index(drop=True)
    test_df = pd.concat(test_frames).reset_index(drop=True)

    x_train = train_df.drop(columns=[config.target_column]).values
    y_train = train_df[config.target_column].values
    x_test = test_df.drop(columns=[config.target_column]).values
    y_test = test_df[config.target_column].values

    # Re-encode to continuous 0-based classes
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    train_df[config.target_column] = y_train
    test_df[config.target_column] = y_test

    return x_train, x_test, y_train, y_test, train_df, test_df


class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNClassifier, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((input_size // 2) * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Expect x to be (batch_size, input_size)
        # Reshape to (batch_size, 1, input_size)
        x = x.view(-1, 1, self.input_size)
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


def train_discriminator(discriminator, optimizer, real_data, fake_data, y_real, loss_fn, penalty_coef):
    N = real_data.size(0)
    optimizer.zero_grad()

    # Real data loss
    prediction_real = discriminator(real_data)
    loss_real = loss_fn(prediction_real.squeeze(),
                        true_target(y_real).squeeze())

    # Fake data loss (không dùng retain_graph)
    prediction_fake = discriminator(fake_data.detach())  # Đảm bảo detach
    loss_fake = loss_fn(prediction_fake, zeros_target(N))

    # Penalty term
    penalty = penalty_coef * torch.mean((prediction_fake - 0.5) ** 2)

    # Total loss
    total_loss = loss_real + loss_fake + penalty
    total_loss.backward()

    # Store gradients if needed
    grads = {n: p.grad.clone() if p.grad is not None else None
             for n, p in discriminator.named_parameters()}

    optimizer.step()

    return {
        'error_real': loss_real.item(),
        'error_fake': loss_fake.item(),
        'penalty': penalty.item(),
        'total_error': total_loss.item(),
        'prediction_real': prediction_real.mean().item(),
        'prediction_fake': prediction_fake.mean().item(),
        'gradients': grads
    }


def train_generator(discriminator, generator, optimizer, fake_data, loss_fn, penalty_coef):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    loss = loss_fn(prediction, ones_target(N))
    penalty = penalty_coef * torch.mean((prediction - 1.0) ** 2)
    total_loss = loss + penalty
    total_loss.backward()
    grads = {n: p.grad.clone()
             for n, p in generator.named_parameters() if p.grad is not None}
    optimizer.step()
    return {
        'error': loss.item(),
        'total_error': total_loss.item(),
        'gradients': grads
    }


def run_gan_training(config, x_data, y_data, target_class, num_samples, save_log=True):
    data_loader = create_dataloader(
        x_data, (y_data == target_class).astype(int), config.batch_size)
    # Set input_size based on dataset type
    input_size = 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45
    D = DiscriminatorNet(input_size=input_size)
    G = GeneratorNet(output_size=input_size)
    if config.disc_params:
        D.load_state_dict(config.disc_params)
    if config.gen_params:
        G.load_state_dict(config.gen_params)
    d_opt = optim.Adam(D.parameters(), lr=config.lr_disc)
    g_opt = optim.Adam(G.parameters(), lr=config.lr_gen)
    loss_fn = nn.BCELoss()
    history = {'disc_losses': [], 'gen_losses': [],
               'disc_feedback': [], 'gen_feedback': []}

    for epoch in range(config.local_epochs):
        disc_epoch, gen_epoch = [], []
        for real_batch, y_real in data_loader:
            N = real_batch.size(0)
            real_data = real_batch
            fake_data = G(noise(N)).detach()
            y_real = y_real.numpy()
            disc_fb = train_discriminator(
                D, d_opt, real_data, fake_data, y_real, loss_fn, config.penalty_coef)
            gen_fb = train_generator(D, G, g_opt, G(
                noise(N)), loss_fn, config.penalty_coef)
            disc_epoch.append(disc_fb)
            gen_epoch.append(gen_fb)
        history['disc_losses'].append(
            sum(f['total_error'] for f in disc_epoch))
        history['gen_losses'].append(sum(f['total_error'] for f in gen_epoch))
        history['disc_feedback'].append(disc_epoch)
        history['gen_feedback'].append(gen_epoch)
        if epoch % 5 == 0:
            print(
                f"Class {target_class} - Epoch {epoch} | D Loss: {history['disc_losses'][-1]:.4f} | G Loss: {history['gen_losses'][-1]:.4f}")

    x_syn = G(noise(num_samples)).detach().numpy()
    y_syn = np.ones(num_samples) * target_class
    if save_log:
        with open(f"gan_results_class_{target_class}.txt", "w") as f:
            for epoch in range(config.local_epochs):
                f.write(f"Epoch {epoch+1}\n")
                f.write("Discriminator Feedback:\n")
                for i, fb in enumerate(history['disc_feedback'][epoch]):
                    f.write(f"  Batch {i+1}: {fb}\n")
                f.write("Generator Feedback:\n")
                for i, fb in enumerate(history['gen_feedback'][epoch]):
                    f.write(f"  Batch {i+1}: {fb}\n")
                f.write("\n")

    return {
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
        'synthetic_data': (x_syn, y_syn),
        'history': history
    }


def train_ids_model(x_train, y_train, x_test, y_test, config):
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_loader = create_dataloader(
        x_train_tensor, y_train_tensor, config.batch_size)
    test_loader = create_dataloader(
        x_test_tensor, y_test_tensor, config.batch_size)

    num_classes = len(np.unique(y_train))  # 10 for UNSW-NB15
    # Set input_size based on dataset type
    input_size = 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45
    model = CNNClassifier(input_size=input_size, num_classes=num_classes)
    print(
        f"Initialized CNNClassifier with input_size={input_size}, num_classes={num_classes}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {'train_losses': [], 'test_losses': [], 'train_accuracies': [
    ], 'test_accuracies': [], 'f1_scores': [], 'reports': []}

    for epoch in range(config.global_epochs):
        model.train()
        correct, total, loss_val = 0, 0, 0
        for batch_idx, (x, target) in enumerate(train_loader):
            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1}: x.shape={x.shape}, target.shape={target.shape}")
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += x.size(0)
            loss_val = loss_val * 0.9 + loss.item() * 0.1
        train_acc = correct / total
        history['train_losses'].append(loss_val)
        history['train_accuracies'].append(train_acc)
        print(
            f"Epoch {epoch+1} - Train Loss: {loss_val:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        correct, total, loss_val = 0, 0, 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                print(
                    f"Test Batch {batch_idx+1}: x.shape={x.shape}, target.shape={target.shape}")
                out = model(x)
                loss = criterion(out, target)
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += x.size(0)
                loss_val = loss_val * 0.9 + loss.item() * 0.1
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                try:
                    f1 = f1_score(target.cpu().numpy(),
                                  pred.cpu().numpy(), average=None)
                    history['f1_scores'].append(f1)
                    history['reports'].append(classification_report(
                        target.cpu().numpy(), pred.cpu().numpy(), zero_division=0, output_dict=True))
                except Exception as e:
                    print(f"Warning: Could not calculate F1 scores - {e}")

        test_acc = correct / total
        history['test_losses'].append(loss_val)
        history['test_accuracies'].append(test_acc)
        print(
            f"Epoch {epoch+1} - Test Loss: {loss_val:.4f}, Test Acc: {test_acc:.4f}")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    try:
        final_report = classification_report(
            all_targets, all_preds, zero_division=0)
        print(final_report)
    except Exception as e:
        print(f"Warning: Could not generate final classification report - {e}")
        final_report = "Classification report generation failed."

    return {
        'model': model.state_dict(),
        'final_report': final_report,
        'history': history,
        'predictions': all_preds,
        'true_labels': all_targets
    }


def run_gan_ids_pipeline(config, verbose=True):
    if verbose:
        print("=== Starting GAN-IDS Pipeline ===")
        print(f"Dataset: {config.dataset_type}")
        print(f"Labels to consider: {config.labels_to_consider}")
        print(f"F1 Threshold: {config.f1_threshold}")

    try:
        # 1. Load and preprocess data
        if verbose:
            print("\n1. Loading and preprocessing data...")
        x_train, x_test, y_train, y_test, train_df, test_df = load_and_preprocess_data(
            config)

        # 2. Train initial IDS model
        if verbose:
            print("\n2. Training initial IDS model...")
        initial_results = train_ids_model(
            x_train, y_train, x_test, y_test, config)

        # 3. Identify weak classes
        if verbose:
            print("\n3. Identifying weak performing classes...")
        if not initial_results['history']['reports']:
            print(
                "Warning: No classification reports generated. Computing metrics directly.")
            y_pred = initial_results['predictions']
            y_true = initial_results['true_labels']
            f1_per_class = f1_score(y_true, y_pred, average=None)
            final_report = {str(i): {'f1-score': f1}
                            for i, f1 in enumerate(f1_per_class)}
        else:
            final_report = initial_results['history']['reports'][-1]

        weak_classes, normal_classes = [], []
        for class_str, metrics in final_report.items():
            if not isinstance(metrics, dict):
                continue
            try:
                cls_id = int(class_str)
                f1 = metrics.get('f1-score', 1.0)
                if f1 < config.f1_threshold:
                    weak_classes.append(cls_id)
                else:
                    normal_classes.append(cls_id)
            except (ValueError, KeyError):
                continue

        if verbose:
            print(f"Weak classes (F1 < {config.f1_threshold}): {weak_classes}")
            print(f"Normal classes: {normal_classes}")

        # 4. Generate synthetic data with GAN
        if verbose:
            print("\n4. Generating synthetic data with GAN...")
        gan_results = {}
        synthetic_data_all = []
        unique_classes = np.unique(np.concatenate([y_train, y_test]))

        for cls in unique_classes:
            samples = config.num_samples_weak if cls in weak_classes else config.num_samples_normal
            if verbose:
                print(f"   Class {cls}: {samples} samples")
            cls_mask = (train_df[config.target_column] == cls)
            if cls_mask.sum() < 10:
                if verbose:
                    print(
                        f"   Skipping class {cls}: too few samples ({cls_mask.sum()})")
                continue
            x_cls = train_df[cls_mask].drop(
                columns=[config.target_column]).values
            y_cls = train_df[cls_mask][config.target_column].values
            try:
                gan_output = run_gan_training(
                    config, x_cls, y_cls, cls, samples, save_log=False)
                synthetic_data_all.append(gan_output['synthetic_data'])
                gan_results[cls] = gan_output
            except Exception as e:
                if verbose:
                    print(f"   Error training GAN for class {cls}: {e}")
                continue

        # Helper function to reshape synthetic data to 4D
        def reshape_synthetic_data(x_syn, config):
            if config.dataset_type in ['KDD99', 'NSL-KDD']:
                shape = (7, 6)
            else:  # UNSW-NB15
                shape = (9, 5)
            padded = np.zeros((x_syn.shape[0], shape[0] * shape[1]))
            padded[:, :min(x_syn.shape[1], shape[0] * shape[1])
                   ] = x_syn[:, :min(x_syn.shape[1], shape[0] * shape[1])]
            return padded.reshape((-1, shape[0], shape[1], 1))

        # 5. Augment training data
        if verbose:
            print("\n5. Augmenting training data with synthetic samples...")
        if synthetic_data_all:
            for x_syn, y_syn in synthetic_data_all:
                # Reshape x_syn to match x_train's 4D shape
                x_syn_reshaped = reshape_synthetic_data(x_syn, config)
                x_train = np.concatenate((x_train, x_syn_reshaped), axis=0)
                y_train = np.concatenate((y_train, y_syn), axis=0)
            if verbose:
                print(f"   Augmented training set size: {x_train.shape}")
        else:
            if verbose:
                print("   No synthetic data was generated, using original training data")

        # 6. Train augmented IDS model
        if verbose:
            print("\n6. Training augmented IDS model...")
        augmented_results = train_ids_model(
            x_train, y_train, x_test, y_test, config)

        # 7. Compare results
        if verbose:
            print("\n=== Results Summary ===")
            f1_initial = initial_results['history']['f1_scores'][-1] if initial_results['history']['f1_scores'] else f1_score(
                initial_results['true_labels'], initial_results['predictions'], average=None)
            f1_augmented = augmented_results['history']['f1_scores'][-1] if augmented_results['history']['f1_scores'] else f1_score(
                augmented_results['true_labels'], augmented_results['predictions'], average=None)
            print(f"Average F1 - Initial: {np.mean(f1_initial):.4f}")
            print(f"Average F1 - Augmented: {np.mean(f1_augmented):.4f}")
            print(
                f"Improvement: {np.mean(f1_augmented) - np.mean(f1_initial):+.4f}")

        return {
            'initial_ids': initial_results,
            'augmented_ids': augmented_results,
            'gan': gan_results,
            'normal_classes': normal_classes,
            'weak_classes': weak_classes,
            'dataset_info': {
                'train_shape': x_train.shape,
                'test_shape': x_test.shape,
                'num_classes': len(unique_classes)
            }
        }

    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        raise


def plot_f1_scores(initial_f1, augmented_f1, save_path_prefix='f1_comparison'):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_path_prefix}_{timestamp}.png"

    x = np.arange(len(initial_f1))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, initial_f1, width, label='Initial')
    plt.bar(x + width/2, augmented_f1, width, label='Augmented')
    plt.axhline(y=0.95, color='r', linestyle='--', label='F1 Threshold')
    plt.xlabel('Class Index')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison per Class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved F1 comparison plot to {save_path}")


def save_model(model_state, filename):
    torch.save(model_state, filename)
    print(f"Model saved to {filename}")

# 3. Improved evaluation function với error handling


def evaluate_model_performance(model, test_loader, device='cpu'):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            try:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(
        all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions,
                          average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions,
                  average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_loss': total_loss / len(test_loader),
        'predictions': all_predictions,
        'targets': all_targets,
        'classification_report': classification_report(all_targets, all_predictions, zero_division=0)
    }


def evaluate_stability(config, n_runs=5):
    avg_improvements = []
    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} ===")
        results = run_gan_ids_pipeline(config)
        f1_initial = results['initial_ids']['history']['f1_scores'][-1]
        f1_augmented = results['augmented_ids']['history']['f1_scores'][-1]
        improvement = np.mean(f1_augmented) - np.mean(f1_initial)
        print(f"Run {i+1} Improvement: {improvement:.4f}")
        avg_improvements.append(improvement)
    print(
        f"\n>>> Average Improvement over {n_runs} runs: {np.mean(avg_improvements):.4f} ± {np.std(avg_improvements):.4f}")


# === MAIN ===
if __name__ == '__main__':

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/run_log_{timestamp}.txt"

    class Logger(object):
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = sys.stderr = Logger(log_filename)
    print(f"===> Logging to {log_filename}")

    try:
        # Initialize configuration
        config = GANIDSConfig()

        config.prompt_dataset_type()  # ← Gọi hàm yêu cầu nhập dữ liệu

        # Validate configuration
        validate_config(config)

        # Option to override default parameters
        """
        config.set_training_params({
            'local_epochs': 20,
            'global_epochs': 50,  # Reduced for testing
            'batch_size': 500,
            'pca_components': 20,
            'f1_threshold': 0.9,
            'dataset_type': 'KDD99'
        })
        """

        # Run the pipeline
        results = run_gan_ids_pipeline(config, verbose=True)

        # Generate plots and save results

        print("Len f1_scores (initial):", len(
            results['initial_ids']['history']['f1_scores']))
        print("Len f1_scores (augmented):", len(
            results['augmented_ids']['history']['f1_scores']))

        try:
            f1_initial_scores = results['initial_ids']['history']['f1_scores']
            f1_initial = f1_initial_scores[-1] if f1_initial_scores else 0
        except Exception as e:
            print(f"Warning: f1_scores for initial_ids error: {e}")
            f1_initial = 0

        try:
            f1_augmented_scores = results['augmented_ids']['history']['f1_scores']
            f1_augmented = f1_augmented_scores[-1] if f1_augmented_scores else 0
        except Exception as e:
            print(f"Warning: f1_scores for augmented_ids error: {e}")
            f1_augmented = 0

        plot_f1_scores(f1_initial, f1_augmented)
        save_model(results['augmented_ids']['model'],
                   'augmented_ids_model.pth')

        print("\n=== Pipeline completed successfully ===")

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
