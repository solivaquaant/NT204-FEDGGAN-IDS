import socket
import pickle
import struct
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

client_id=0
port=8888


class ACGANIDSConfig:
    def __init__(self):
        self.gen_params = None
        self.disc_params = None
        self.local_epochs = 1
        self.global_epochs = 1
        self.batch_size = 5000
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


def load_and_preprocess_data(config: ACGANIDSConfig):
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
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label' #'drop'
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

        # Define reshape dimensions for 2D CNN
        if config.dataset_type in ['KDD99', 'NSL-KDD']:
            shape = (7, 6, 1)  # 2D shape with channel dimension
        else:
            shape = (9, 5, 1)  # 2D shape with channel dimension

        # Extract and reshape features for 2D CNN
        def reshape_features_2d(X, shape):
            padded = np.zeros((X.shape[0], shape[0] * shape[1]))
            padded[:, :min(X.shape[1], shape[0] * shape[1])] = X[:,
                                                                 :min(X.shape[1], shape[0] * shape[1])]
            return padded.reshape((-1, shape[2], shape[0], shape[1]))  # Channel first format for PyTorch

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
        x_train_reshaped = reshape_features_2d(X_train, shape)
        x_test_reshaped = reshape_features_2d(X_test, shape)

        # Create DataFrames with reshaped features
        feature_columns = [f'feature_{i}' for i in range(
            x_train_reshaped.shape[1] * x_train_reshaped.shape[2] * x_train_reshaped.shape[3])]
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


def noise(size, latent_dim=100):
    """Generate random noise for ACGAN generator"""
    return torch.randn(size, latent_dim)


def ones_target(size):
    return torch.ones(size, 1)


def zeros_target(size):
    return torch.zeros(size, 1)


def true_target(y):
    return torch.tensor(y, dtype=torch.float32).view(-1, 1)


def create_dataloader(x, y, batch_size, shuffle=True):
    # Handle 4D input - keep the 2D structure
    print(f"Input shape: {x.shape}")

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


def prepare_datasets(dataset: pd.DataFrame, config: ACGANIDSConfig):
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


class CNN2DClassifier(nn.Module):
    """2D CNN Classifier for reshaped network traffic data"""
    def __init__(self, input_channels, height, width, num_classes):
        super(CNN2DClassifier, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, 
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # Calculate the flattened size after convolutions
        # After conv1 + pool1: (height-1) x (width-1)
        # After conv2 + pool2: (height-2) x (width-2)
        conv_output_height = max(1, height - 2)
        conv_output_width = max(1, width - 2)
        self.flattened_size = 64 * conv_output_height * conv_output_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ACGANDiscriminator(nn.Module):
    """ACGAN Discriminator - outputs both real/fake and class predictions"""
    def __init__(self, input_channels, height, width, num_classes):
        super(ACGANDiscriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=1)
        
        # Calculate flattened size
        conv_output_height = max(1, height - 2)
        conv_output_width = max(1, width - 2)
        self.flattened_size = 64 * conv_output_height * conv_output_width
        
        # Shared layers
        self.fc_shared = nn.Linear(self.flattened_size, 128)
        
        # Real/fake discriminator head
        self.fc_discriminator = nn.Linear(128, 1)
        
        # Class classifier head
        self.fc_classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc_shared(x))
        
        # Real/fake output
        discriminator_output = torch.sigmoid(self.fc_discriminator(x))
        
        # Class output
        class_output = self.fc_classifier(x)
        
        return discriminator_output, class_output


class ACGANGenerator(nn.Module):
    """ACGAN Generator - takes noise and class label as input"""
    def __init__(self, latent_dim, num_classes, output_channels, height, width):
        super(ACGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.height = height
        self.width = width
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        self.fc1 = nn.Linear(latent_dim * 2, 128)  # *2 because we concat noise and class embedding
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_channels * height * width)
        
        self.tanh = nn.Tanh()

    def forward(self, noise, labels):
        # Embed class labels
        class_embed = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        x = torch.cat([noise, class_embed], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        
        # Reshape to image format
        x = x.view(-1, self.output_channels, self.height, self.width)
        return x


def get_gradients(model):
    """Trả về dict gradients của model hiện tại."""
    return {n: p.grad.clone().cpu() if p.grad is not None else torch.zeros_like(p.data).cpu() for n, p in model.named_parameters()}


def train_acgan_discriminator(discriminator, optimizer, real_data, fake_data, real_labels, fake_labels, real_validity, fake_validity, loss_fn_adv, loss_fn_cls, penalty_coef):
    """Train ACGAN discriminator"""
    optimizer.zero_grad()
    
    # Real data
    real_validity_pred, real_class_pred = discriminator(real_data)
    real_adv_loss = loss_fn_adv(real_validity_pred.squeeze(), real_validity.squeeze())
    real_cls_loss = loss_fn_cls(real_class_pred, real_labels)
    
    # Fake data
    fake_validity_pred, fake_class_pred = discriminator(fake_data.detach())
    fake_adv_loss = loss_fn_adv(fake_validity_pred.squeeze(), fake_validity.squeeze())
    fake_cls_loss = loss_fn_cls(fake_class_pred, fake_labels)
    
    # Total discriminator loss
    d_loss = (real_adv_loss + fake_adv_loss) / 2 + (real_cls_loss + fake_cls_loss) / 2
    
    # Add penalty
    penalty = penalty_coef * torch.mean((fake_validity_pred - 0.5) ** 2)
    total_loss = d_loss + penalty
    
    total_loss.backward()
    grads = get_gradients(discriminator)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'd_loss': d_loss.item(),
        'real_adv_loss': real_adv_loss.item(),
        'fake_adv_loss': fake_adv_loss.item(),
        'real_cls_loss': real_cls_loss.item(),
        'fake_cls_loss': fake_cls_loss.item(),
        'penalty': penalty.item(),
        'gradients': grads
    }


def train_acgan_generator(generator, discriminator, optimizer, noise_batch, labels_batch, loss_fn_adv, loss_fn_cls, penalty_coef):
    """Train ACGAN generator"""
    optimizer.zero_grad()
    
    # Generate fake data
    fake_data = generator(noise_batch, labels_batch)
    
    # Get discriminator predictions
    fake_validity_pred, fake_class_pred = discriminator(fake_data)
    
    # Generator wants discriminator to classify fake data as real
    target_validity = torch.ones_like(fake_validity_pred)
    
    # Adversarial loss
    adv_loss = loss_fn_adv(fake_validity_pred.squeeze(), target_validity.squeeze())
    
    # Classification loss  
    cls_loss = loss_fn_cls(fake_class_pred, labels_batch)
    
    # Total generator loss
    g_loss = adv_loss + cls_loss
    
    # Add penalty
    penalty = penalty_coef * torch.mean((fake_validity_pred - 1.0) ** 2)
    total_loss = g_loss + penalty
    
    total_loss.backward()
    grads = get_gradients(generator)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'g_loss': g_loss.item(),
        'adv_loss': adv_loss.item(),
        'cls_loss': cls_loss.item(),
        'penalty': penalty.item(),
        'gradients': grads
    }


def run_acgan_training(config, x_data, y_data, target_class, num_samples, save_log=True):
    """Train ACGAN for specific class"""
    
    # Filter data for target class
    class_mask = (y_data == target_class)
    x_class = x_data[class_mask]
    y_class = y_data[class_mask]
    
    if len(x_class) == 0:
        print(f"Warning: No samples found for class {target_class}")
        return None
    
    print(f"Training ACGAN for class {target_class} with {len(x_class)} samples")
    
    # Create dataloader
    data_loader = create_dataloader(x_class, y_class, config.batch_size)
    
    # Get dimensions from data shape
    if len(x_data.shape) == 4:  # (N, C, H, W)
        input_channels, height, width = x_data.shape[1], x_data.shape[2], x_data.shape[3]
    else:
        raise ValueError(f"Expected 4D input data, got shape: {x_data.shape}")
    
    num_classes = len(np.unique(y_data))
    latent_dim = 100
    
    # Initialize models
    generator = ACGANGenerator(latent_dim, num_classes, input_channels, height, width)
    discriminator = ACGANDiscriminator(input_channels, height, width, num_classes)
    
    # Load existing weights if available
    if config.gen_params:
        generator.load_state_dict(config.gen_params)
    if config.disc_params:
        discriminator.load_state_dict(config.disc_params)
    
    # Optimizers
    d_opt = optim.Adam(discriminator.parameters(), lr=config.lr_disc)
    g_opt = optim.Adam(generator.parameters(), lr=config.lr_gen)
    
    # Loss functions
    loss_fn_adv = nn.BCELoss()
    loss_fn_cls = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'disc_losses': [], 'gen_losses': [], 
        'disc_feedback': [], 'gen_feedback': []
    }
    
    last_gen_grads = None
    last_disc_grads = None
    last_gen_error = None
    last_disc_error = None
    
    for epoch in range(config.local_epochs):
        disc_epoch, gen_epoch = [], []
        
        for batch_idx, (real_batch, real_labels) in enumerate(data_loader):
            batch_size = real_batch.size(0)
            
            # Generate random noise and labels for fake data
            noise_batch = noise(batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (batch_size,))
            
            # Generate fake data
            fake_batch = generator(noise_batch, fake_labels)
            
            # Create validity targets
            real_validity = torch.ones(batch_size, 1)
            fake_validity = torch.zeros(batch_size, 1)
            
            # Train discriminator
            disc_fb = train_acgan_discriminator(
                discriminator, d_opt, real_batch, fake_batch,
                real_labels, fake_labels, real_validity, fake_validity,
                loss_fn_adv, loss_fn_cls, config.penalty_coef
            )
            
            # Train generator
            noise_batch = noise(batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (batch_size,))
            
            gen_fb = train_acgan_generator(
                generator, discriminator, g_opt, noise_batch, fake_labels,
                loss_fn_adv, loss_fn_cls, config.penalty_coef
            )
            
            disc_epoch.append(disc_fb)
            gen_epoch.append(gen_fb)
            
            # Store last gradients and errors
            last_disc_grads = disc_fb['gradients']
            last_disc_error = disc_fb['total_loss']
            last_gen_grads = gen_fb['gradients']
            last_gen_error = gen_fb['total_loss']
        
        # Calculate epoch losses
        disc_loss = np.mean([f['total_loss'] for f in disc_epoch])
        gen_loss = np.mean([f['total_loss'] for f in gen_epoch])
        
        history['disc_losses'].append(disc_loss)
        history['gen_losses'].append(gen_loss)
        history['disc_feedback'].append(disc_epoch)
        history['gen_feedback'].append(gen_epoch)
        
        if epoch % 5 == 0:
            print(f"Class {target_class} - Epoch {epoch} | D Loss: {disc_loss:.4f} | G Loss: {gen_loss:.4f}")
    
    # Generate synthetic data
    with torch.no_grad():
        syn_noise = noise(num_samples, latent_dim)
        syn_labels = torch.full((num_samples,), target_class, dtype=torch.long)
        x_syn = generator(syn_noise, syn_labels).detach().numpy()
        y_syn = np.full(num_samples, target_class)
    
    # Save training log
    if save_log:
        with open(f"acgan_results_class_{target_class}.txt", "w") as f:
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
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'synthetic_data': (x_syn, y_syn),
        'history': history,
        'gen_grads': last_gen_grads,
        'disc_grads': last_disc_grads,
        'gen_error': last_gen_error,
        'disc_error': last_disc_error
    }


def train_ids_model(x_train, y_train, x_test, y_test, config):
    """Train 2D CNN IDS model"""
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Convert to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_loader = create_dataloader(x_train_tensor, y_train_tensor, config.batch_size)
    test_loader = create_dataloader(x_test_tensor, y_test_tensor, config.batch_size, shuffle=False)

    # Model parameters
    num_classes = len(np.unique(y_train))
    if len(x_train.shape) == 4:  # (N, C, H, W)
        input_channels, height, width = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    else:
        raise ValueError(f"Expected 4D input data, got shape: {x_train.shape}")
    
    # Initialize 2D CNN model
    model = CNN2DClassifier(input_channels, height, width, num_classes)
    print(f"Initialized CNN2DClassifier with input_channels={input_channels}, height={height}, width={width}, num_classes={num_classes}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_losses': [], 'test_losses': [], 
        'train_accuracies': [], 'test_accuracies': [], 
        'f1_scores': [], 'reports': []
    }

    for epoch in range(config.global_epochs):
        # Training phase
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += x.size(0)
            running_loss += loss.item()
        
        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader)
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Testing phase
        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                out = model(x)
                loss = criterion(out, target)
                pred = out.argmax(dim=1)
                
                correct += (pred == target).sum().item()
                total += x.size(0)
                running_loss += loss.item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_acc = correct / total
        avg_test_loss = running_loss / len(test_loader)
        history['test_losses'].append(avg_test_loss)
        history['test_accuracies'].append(test_acc)
        
        print(f"Epoch {epoch+1} - Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Calculate F1 scores
        try:
            f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
            history['f1_scores'].append(f1)
            history['reports'].append(classification_report(
                all_targets, all_preds, zero_division=0, output_dict=True))
        except Exception as e:
            print(f"Warning: Could not calculate F1 scores - {e}")

    # Final evaluation
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    try:
        final_report = classification_report(all_targets, all_preds, zero_division=0)
        print("Final Classification Report:")
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


def receive_config_from_server(client_socket):
    """Receive configuration from server"""
    config_dict = receive_data(client_socket)
    config = ACGANIDSConfig()
    config.set_training_params(config_dict)
    return config


def train_acgan_local(config, x_train, y_train):
    """Train ACGAN locally for all classes"""
    print("Training local ACGAN for all classes...")
    
    # Get data dimensions
    if len(x_train.shape) == 4:  # (N, C, H, W)
        input_channels, height, width = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    else:
        raise ValueError(f"Expected 4D input data, got shape: {x_train.shape}")
    
    num_classes = len(np.unique(y_train))
    latent_dim = 100
    
    # Initialize models
    generator = ACGANGenerator(latent_dim, num_classes, input_channels, height, width)
    discriminator = ACGANDiscriminator(input_channels, height, width, num_classes)
    
    # Train for a few epochs locally
    data_loader = create_dataloader(x_train, y_train, config.batch_size)
    
    d_opt = optim.Adam(discriminator.parameters(), lr=config.lr_disc)
    g_opt = optim.Adam(generator.parameters(), lr=config.lr_gen)
    
    loss_fn_adv = nn.BCELoss()
    loss_fn_cls = nn.CrossEntropyLoss()
    
    print(f"Training ACGAN locally for {config.local_epochs} epochs...")
    
    for epoch in range(min(config.local_epochs, 3)):  # Limited local training
        for batch_idx, (real_batch, real_labels) in enumerate(data_loader):
            batch_size = real_batch.size(0)
            
            # Generate fake data
            noise_batch = noise(batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (batch_size,))
            fake_batch = generator(noise_batch, fake_labels)
            
            # Create validity targets
            real_validity = torch.ones(batch_size, 1)
            fake_validity = torch.zeros(batch_size, 1)
            
            # Train discriminator
            train_acgan_discriminator(
                discriminator, d_opt, real_batch, fake_batch,
                real_labels, fake_labels, real_validity, fake_validity,
                loss_fn_adv, loss_fn_cls, config.penalty_coef
            )
            
            # Train generator
            noise_batch = noise(batch_size, latent_dim)
            fake_labels = torch.randint(0, num_classes, (batch_size,))
            
            train_acgan_generator(
                generator, discriminator, g_opt, noise_batch, fake_labels,
                loss_fn_adv, loss_fn_cls, config.penalty_coef
            )
        
        if epoch % 1 == 0:
            print(f"Local ACGAN training - Epoch {epoch+1} completed")
    
    return generator, discriminator


def run_acgan_ids_pipeline_from_server(verbose=True, client_id=0, host='127.0.0.1', port=8888):
    """Run complete ACGAN-IDS pipeline with server communication"""
    if verbose:
        print("=== Starting FEDACGAN-IDS Client Pipeline ===")
        print(f"Client ID: {client_id}")
    
    try:
        # Connect to server
        client_socket = connect_to_server(host, port)
        if verbose:
            print(f"Connected to server at {host}:{port}")
        
        # 1. Receive config from server
        if verbose:
            print("1. Receiving config from server...")
        config = receive_config_from_server(client_socket)
        if verbose:
            print(f"Received config for dataset: {config.dataset_type}")
        
        # 2. Load and preprocess data
        if verbose:
            print("2. Loading and preprocessing data...")
        x_train, x_test, y_train, y_test, train_df, test_df = load_and_preprocess_data(config)
        
        # 3. Train initial IDS model
        if verbose:
            print("3. Training initial IDS model (before ACGAN augmentation)...")
        initial_results = train_ids_model(x_train, y_train, x_test, y_test, config)
        f1_initial = initial_results['history']['f1_scores'][-1] if initial_results['history']['f1_scores'] else \
                   f1_score(initial_results['true_labels'], initial_results['predictions'], average=None, zero_division=0)
        print("Initial IDS F1-score per class:", f1_initial)
        
        # 4. Train local ACGAN
        if verbose:
            print("4. Training local ACGAN...")
        generator, discriminator = train_acgan_local(config, x_train, y_train)
        
        # 5. Send local ACGAN weights to server
        if verbose:
            print("5. Sending local ACGAN weights to server...")
        send_data(client_socket, {
            'client_id': client_id,
            'gen_state': generator.state_dict(),
            'disc_state': discriminator.state_dict(),
            'data_shape': x_train.shape,
            'num_classes': len(np.unique(y_train)),
            'input_size': np.prod(x_train.shape[1:])  # thêm dòng này
        })
        
        # 6. Receive updated weights from server
        if verbose:
            print("6. Receiving updated ACGAN weights from server...")
        response = receive_data(client_socket)
        new_gen_weights = response['new_gen_weights']
        new_disc_weights = response['new_disc_weights']
        
        # 7. Update local ACGAN with new weights
        if verbose:
            print("7. Updating local ACGAN with federated weights...")
        generator.load_state_dict(new_gen_weights)
        discriminator.load_state_dict(new_disc_weights)
        
        # 8. Generate synthetic data for augmentation
        if verbose:
            print("8. Generating synthetic data for augmentation...")
        
        synthetic_data = []
        synthetic_labels = []
        
        unique_classes = np.unique(y_train)
        samples_per_class = config.num_samples_normal
        
        with torch.no_grad():
            for class_label in unique_classes:
                # Generate synthetic samples for each class
                latent_dim = 100
                syn_noise = noise(samples_per_class, latent_dim)
                syn_labels = torch.full((samples_per_class,), class_label, dtype=torch.long)
                syn_data = generator(syn_noise, syn_labels).detach().numpy()
                
                synthetic_data.append(syn_data)
                synthetic_labels.extend([class_label] * samples_per_class)
        
        # Combine synthetic data
        x_synthetic = np.vstack(synthetic_data)
        y_synthetic = np.array(synthetic_labels)
        
        # Augment training data
        x_augmented = np.vstack([x_train, x_synthetic])
        y_augmented = np.hstack([y_train, y_synthetic])
        
        print(f"Original training data: {x_train.shape}")
        print(f"Synthetic data: {x_synthetic.shape}")
        print(f"Augmented training data: {x_augmented.shape}")
        
        # 9. Train IDS model with augmented data
        if verbose:
            print("9. Training IDS model with augmented data...")
        augmented_results = train_ids_model(x_augmented, y_augmented, x_test, y_test, config)
        f1_augmented = augmented_results['history']['f1_scores'][-1] if augmented_results['history']['f1_scores'] else \
                     f1_score(augmented_results['true_labels'], augmented_results['predictions'], average=None, zero_division=0)
        print("Augmented IDS F1-score per class:", f1_augmented)
        
        # 10. Plot comparison
        if verbose:
            print("10. Plotting F1-score comparison...")
        plot_f1_scores(f1_initial, f1_augmented, dataset_name=config.dataset_type, client_id=client_id)
        
        # Close connection
        client_socket.close()
        
        if verbose:
            print("Pipeline completed successfully!")
        
        return {
            'initial_results': initial_results,
            'augmented_results': augmented_results,
            'f1_initial': f1_initial,
            'f1_augmented': f1_augmented,
            'improvement': np.mean(f1_augmented) - np.mean(f1_initial)
        }
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        if 'client_socket' in locals():
            client_socket.close()
        raise


def plot_f1_scores(initial_f1, augmented_f1, dataset_name=None, client_id=None):
    """Plot F1 score comparison"""
    min_len = min(len(initial_f1), len(augmented_f1))
    x = np.arange(min_len)
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, initial_f1[:min_len], width, label='Before ACGAN', alpha=0.8)
    plt.bar(x + width/2, augmented_f1[:min_len], width, label='After ACGAN', alpha=0.8)

    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    title = "F1 Score Comparison: Before vs After ACGAN Augmentation"
    if dataset_name or client_id is not None:
        title += f"\nDataset: {dataset_name} | Client: {client_id}"
    plt.title(title)
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'f1_scores_{dataset_name}_client{client_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"F1 score comparison saved to {filename}")
    plt.show()


def save_model(model_state, filename):
    """Save model state to file"""
    torch.save(model_state, filename)
    print(f"Model saved to {filename}")


def evaluate_model_performance(model, test_loader, device='cpu'):
    """Comprehensive model evaluation with error handling"""
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
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0) 
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

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
    """Evaluate model stability across multiple runs"""
    avg_improvements = []
    for i in range(n_runs):
        print(f"\n=== Stability Run {i+1}/{n_runs} ===")
        try:
            results = run_acgan_ids_pipeline_from_server(
                verbose=True, client_id=0, host='127.0.0.1', port=8888)
            
            improvement = results['improvement']
            print(f"Run {i+1} Improvement: {improvement:.4f}")
            avg_improvements.append(improvement)
            
        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            continue
    
    if avg_improvements:
        print(f"\n>>> Average Improvement over {len(avg_improvements)} successful runs: {np.mean(avg_improvements):.4f} ± {np.std(avg_improvements):.4f}")
    else:
        print("No successful runs completed.")


def send_data(client_socket, data):
    """Send serialized data to server"""
    serialized_data = pickle.dumps(data)
    client_socket.sendall(struct.pack('!I', len(serialized_data)))
    client_socket.sendall(serialized_data)


def receive_data(client_socket):
    """Receive and deserialize data from server"""
    raw_length = client_socket.recv(4)
    if not raw_length:
        raise ConnectionError("Server closed connection")
    length = struct.unpack('!I', raw_length)[0]
    data = b""
    while len(data) < length:
        packet = client_socket.recv(length - len(data))
        if not packet:
            raise ConnectionError("Incomplete data received")
        data += packet
    return pickle.loads(data)


def connect_to_server(host='127.0.0.1', port=8888):
    """Establish connection to the federated learning server"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        return client_socket
    except ConnectionError as e:
        print(f"Failed to connect to server at {host}:{port}")
        raise


# === MAIN EXECUTION ===
if __name__ == '__main__':
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/acgan_run_log_{timestamp}.txt"

    # Setup logging
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
    print("=== ACGAN-IDS Federated Learning Client ===")

    try:
        # Initialize configuration
        config = ACGANIDSConfig()
        
        # Prompt for dataset type
        config.prompt_dataset_type()
        
        # Validate configuration
        validate_config(config)
        
        print(f"Configuration validated for dataset: {config.dataset_type}")
        print(f"Using ACGAN with CNN2D for improved performance")
        
        # Run the federated ACGAN-IDS pipeline
        results = run_acgan_ids_pipeline_from_server(
            verbose=True, 
            client_id=client_id, 
            host='127.0.0.1', 
            port=port
        )
        
        print("\n=== Final Results ===")
        print(f"F1 Score Improvement: {results['improvement']:.4f}")
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Log saved to: {log_filename}")