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


class GANIDSConfig:
    def __init__(self):
        self.gen_params = None #grad G_l
        self.disc_params = None #grad D_l 
        self.gan_epochs = 1 
        self.ids_epochs = 1
        self.batch_size = 5000 #32 / 64
        # Số lượng dữ liệu GAN sinh ra
        self.num_samples_normal = 10
        self.num_samples_weak = 50
        # Tốc độ học
        self.lr_disc = 0.0002
        self.lr_gen = 0.0002
        self.penalty_coef = 0.1 #Hệ số phạt
        # Thông số train
        self.labels_to_consider = 15 #Lấy 15 nhãn có nhiều dữ liệu nhất
        self.f1_threshold = 0.9 #Dưới 0.9 được xem là lớp yếu 

        # Tham số dataset
        self.dataset_type = 'UNSW-NB15' # Loại
        self.categorical_columns = []

        
        self.target_column_map = { 
            'KDD99': 'label',
            'NSL-KDD': 'label',
            'UNSW-NB15': 'attack_cat'
        } #Map dataset types to their target columns
        self.feature_columns = [] #Thuộc tính

        self.dataset_paths = { 
            'KDD99': 'kdd-cup-1999-data/kddcup.data_10_percent.gz',
            'NSL-KDD': 'nsl-kdd-data/NSL_KDD_Train.csv',
            'UNSW-NB15': 'unsw-nb15/UNSW_NB15_training-set.csv'
        } #Đường dẫn tới dataset
        self.test_dataset_paths = {
            'KDD99': 'kdd-cup-1999-data/kddcup.data_10_percent_corrected',
            'NSL-KDD': 'nsl-kdd-data/NSL_KDD_Test.csv',
            'UNSW-NB15': 'unsw-nb15/UNSW_NB15_testing-set.csv'
        }

        #Set tham số từ server vào config (f_eceive_config_from_server)
    def set_training_params(self, params_dict: Dict[str, Any]) -> None:
        valid_params = [
            'gan_epochs', 'ids_epochs', 'batch_size', 'num_samples_normal', 'num_samples_weak',
            'lr_disc', 'lr_gen', 'penalty_coef',
            'labels_to_consider', 'f1_threshold', 'dataset_type'
        ]
        for param, value in params_dict.items():
            if param in valid_params:
                setattr(self, param, value)
            else:
                warnings.warn(
                    f"Parameter '{param}' is not recognized and will be ignored.")
        # Trả về đường dẫn dataset_train (f_oad_and_preprocess_data)
    def get_dataset_path(self) -> str:
        return self.dataset_paths[self.dataset_type]
        # Trả về đường dẫn dataset_test (f_oad_and_preprocess_data)
    def get_test_dataset_path(self) -> str:
        return self.test_dataset_paths[self.dataset_type]
        # Lấy tên cột (f_oad_and_preprocess_data)
    def get_target_column(self) -> str:
        """Return the target column name for the current dataset type."""
        if self.dataset_type not in self.target_column_map:
            raise ValueError(
                f"No target column defined for dataset type: {self.dataset_type}")
        return self.target_column_map[self.dataset_type]
    
    # Chọn dataset (f_main)
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

# Kiểm tra tính hợp lệ của các tham só (f_main)
def validate_config(config):
    """Validate configuration parameters"""
    errors = []
    if config.labels_to_consider <= 0:
        errors.append("Labels to consider must be positive")
    if not 0 < config.f1_threshold <= 1:
        errors.append("F1 threshold must be between 0 and 1")
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    if config.gan_epochs <= 0 or config.ids_epochs <= 0:
        errors.append("Epochs must be positive")
    if config.dataset_type not in config.dataset_paths:
        errors.append(f"Unsupported dataset type: {config.dataset_type}")
    if errors:
        raise ValueError("Configuration errors:\n" +
                         "\n".join(f"- {error}" for error in errors))
    return True

#Tiền xử lí (f_un_gan_ids_pipeline)
def load_and_preprocess_data(config: GANIDSConfig):
    try:
        # Lấy tham số từ config
        path = config.get_dataset_path() #Đường dẫn dataset_train
        test_path = config.get_test_dataset_path() #Đường dẫn dataset_test
        config.target_column = config.get_target_column() #Cột_target

        # Kiểm tra đường dẫn có tồn tại
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test dataset file not found: {test_path}")

        # Liệt kê các đặc trưng cho từng dataset
        # Load KDD99
        if config.dataset_type == 'KDD99':
            features_kdd = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
            ] #Các đặc trưng: 42
            # Load dữ liệu train
            if path.endswith('.gz'): # Mở file zip 
                with gzip.open(path, 'rt') as f:
                    df = pd.read_csv(f, names=features_kdd, header=None)
            else: # Đọc
                df = pd.read_csv(path, names=features_kdd, header=None)
            config.categorical_columns = ['protocol_type', 'service', 'flag']

            # Load dữ liệu test
            if test_path.endswith('.gz'):
                with gzip.open(test_path, 'rt') as f:
                    df_test = pd.read_csv(f, names=features_kdd, header=None)
            else:
                df_test = pd.read_csv(
                    test_path, names=features_kdd, header=None)
        
        # Load NSL_KDD
        elif config.dataset_type == 'NSL-KDD':
            features_kdd = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label' 
            ] #Các đặc trưng: 42

            # Load data train và test (không cần mở .zip)
            df = pd.read_csv(path, names=features_kdd, header=None)
            df_test = pd.read_csv(test_path, names=features_kdd, header=None)
            config.categorical_columns = ['protocol_type', 'service', 'flag']
        
        # Load UNSW-NB15
        elif config.dataset_type == 'UNSW-NB15':
            # Đọc dữ liệu train, test
            df = pd.read_csv(path)
            df_test = pd.read_csv(test_path)
            config.categorical_columns = [col for col in [
                'proto', 'service', 'state'] if col in df.columns]
        
        # Xử lí lỗi
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_type}")
        # Kiểm tra tính hợp lệ của cột 
        if config.target_column not in df.columns or config.target_column not in df_test.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found in dataset {config.dataset_type}")

        #-------------------------------------
        # Clean data:
        # df: chứa data train, df_test chứa data test (Load dữ liệu) 
        # Xóa các hàng có giá trị thiếu/ trùng lặp, đặt lại index
        df = df.dropna().drop_duplicates().reset_index(drop=True) 
        df_test = df_test.dropna().drop_duplicates().reset_index(drop=True)

        #-------------------------------------
        # Encode categorical columns
        # Chuyển tên cột thành số 
        label_encoders = {}
        for col in config.categorical_columns: #Duyệt qua danh sát cột phân loại (config)
            if col in df.columns and col in df_test.columns:
                le = LabelEncoder() #Tạo đối tượng mới
                
                # Lấy các giá trị unique theo collumn sau khi ghép df và df_test  
                unique_values = pd.concat(
                    [df[col].astype(str), df_test[col].astype(str)]).unique()
                
                le.fit(unique_values) # Đưa nhãn unique_value cho le
                # Đổi tên col thành số trong df, df_test
                df[col] = le.transform(df[col].astype(str)) 
                df_test[col] = le.transform(df_test[col].astype(str)) 
                # Lưu mã hóa 
                label_encoders[col] = le

        #-------------------------------------
        # Normalization:
        # Hàm kiểm tra các  giá trị số trong các feature được chỉ định của mỗi dataset.
        # Nếu hợp lệ, tiến hành chuẩn hóa

        def check_numeric(df, features, dataset_name): #Hàm kiểm tra giá trị là số
            non_numeric = [
                col for col in features if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(
                    f"{dataset_name}: The following columns are not numeric types: {non_numeric}")

        if config.dataset_type in ['KDD99', 'NSL-KDD']: #Các feature chứa số nguyên trong bộ dữ liệu KDD99, NSL-KDD
            numerical_features= [
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
        else: #Các feature chứa số nguyên trong bộ dữ liệu UNSW_NB15
            numerical_features = [
                'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
                'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
                'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                'ct_srv_dst'
            ]

        valid_features = [ 
            col for col in numerical_features if col in df.columns] #lấy danh sách các feature thực tế trong df dựa trên danh sách hợp lệ, data có phải số nguyên?
        check_numeric(df, valid_features, config.dataset_type ) #check data train
        check_numeric(df_test, valid_features, config.dataset_type) #check data test

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler() #Chuẩn hóa min-max
        scaler.fit(df[valid_features]) #Lấy min-max của feature
        df[valid_features] = scaler.transform(df[valid_features]) #data train về [0;1]
        df_test[valid_features] = scaler.transform(df_test[valid_features]) #data test về [0;1]
       
        #-------------------------------------
        #Reshape 2D to 4D
        if config.dataset_type in ['KDD99', 'NSL-KDD']: #shape (dài, rộng)
            shape = (7, 6) # 7*6 = 42
        else: #uwns-nb15
            shape = (9, 5) #9*5 = 45

        def reshape_features(X, shape): #Hàm reshape dữ liệu X
            #Tạo mảng 0 (số mẫu của X, hàng x cột = 42/45)
            padded = np.zeros((X.shape[0], shape[0] * shape[1])) 
            #Chép đặc trưng (thiếu thêm 0, thừa thì bỏ dựa trên shape(hxw))
            padded[:, :min(X.shape[1], shape[0] * shape[1])] = X[:,
                                                                 :min(X.shape[1], shape[0] * shape[1])] 
            return padded.reshape((-1, shape[0], shape[1], 1)) #batch_size, h, w, số kênh ảnh (1: đơn sác)
        
        #-------------------------------------
        # Encode labels
        # I chang Encode categorical columns
        # Loại bỏ nhãn lạ, biến tên nhãn thành số, gán lại trong df
        # config.target_column: Loại tấn công
        
        label_encoder = LabelEncoder() #Tạo đối tượng
        y_train = label_encoder.fit_transform(df[config.target_column]) #Chuyển tất cả nhãn trong df train thành số
        
        test_labels = df_test[config.target_column] #Lấy nhãn từ df test
        valid_mask = test_labels.isin(label_encoder.classes_) #Kiểm tra gán 0/1 nhãn trong test có trong train? 
        
        if not valid_mask.all(): #Nhãn lạ
            unseen_labels = test_labels[~valid_mask].unique() #Thêm vào unseen
            # Loại, in cảnh báo: tên + số lượng
            print(
                f"Warning: Filtering out {sum(~valid_mask)} test samples with unseen labels: {unseen_labels}")
            df_test = df_test[valid_mask].reset_index(drop=True)

        y_test = label_encoder.transform(df_test[config.target_column]) #Mã hóa tên nhãn thành số trong df test

        #-------------------------------------
        # Extract features after filtering
        # Bỏ cột nhãn, còn lại chuyển thành mảng 2D(số mẫu, số đặc trưng)
        X_train = df.drop(columns=[config.target_column]).to_numpy() #train
        X_test = df_test.drop(columns=[config.target_column]).to_numpy() #test

        #Reshape từ 2D -> 4D (gọi hàm f_eshape_features)
        x_train_reshaped = reshape_features(X_train, shape)
        x_test_reshaped = reshape_features(X_test, shape)

        # Tạo DataFrames: chuyển lại thành 2D
            # Tạo cột feature đánh dấu từ 0 -> 42 /45 (range = h x w)
        feature_columns = [f'feature_{i}' for i in range( 
            x_train_reshaped.shape[1] * x_train_reshaped.shape[2])] 
            # Tạo dataframe (batch_size, 42/ 45 (h x w)) vào các cột feature_columns vừa tạo
        train_df = pd.DataFrame(x_train_reshaped.reshape(
            x_train_reshaped.shape[0], -1), columns=feature_columns)
        test_df = pd.DataFrame(x_test_reshaped.reshape(
            x_test_reshaped.shape[0], -1), columns=feature_columns)
            # Thêm cột nhãn tấn công, dạng đã encode
        train_df[config.target_column] = y_train 
        test_df[config.target_column] = y_test

        # In số lượng class trong mỗi dataset
        print(
            f"Training dataset shape: {x_train_reshaped.shape}, Number of Classes: {np.unique(y_train)}")
        print(
            f"Testing dataset shape: {x_test_reshaped.shape}, Number of Classes: {np.unique(y_test)}")
        # In số lượng mẫu trong mỗi class
        print(
            f"Class distribution (train):\n{pd.Series(y_train).value_counts()}")
        print(
            f"Class distribution (test):\n{pd.Series(y_test).value_counts()}")

        return x_train_reshaped, x_test_reshaped, y_train, y_test, train_df, test_df
    # Handle lỗi
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


class Data_Loader(Dataset):
    def __init__(self, data_list): # (img, label)
        self.data = data_list

    def __len__(self): #length
        return len(self.data)

    def __getitem__(self, index): #out: (img, label) [index]
        img = self.data[index][0]

        if isinstance(img, torch.Tensor): # img/ tensor -> float
            img_tensor = img.detach().clone().float()
        else: # numpy -> float
            img_tensor = torch.tensor(img, dtype=torch.float32)

        label = self.data[index][1] # giữ nguyên nhãn
        return (img_tensor, label)


def noise(size): # tạo N tensor nhiễu (row), column = 5 
    return torch.randn(size, 5)


def ones_target(size): # 1: label thật (train G_l)
    return torch.ones(size, 1)


def zeros_target(size): # 0: label giả (train D_l)
    return torch.zeros(size, 1)


def true_target(y): # chuyển mảng nhãn (int -> float) thành tensor 1 cột 
    return torch.tensor(y, dtype=torch.float32).view(-1, 1)


def create_dataloader(x, y, batch_size, shuffle=True):
    # Reshape 4D -> 2D
    if x.ndim == 4:
        x = x.reshape(x.shape[0], -1)  # Flatten to (batch_size, 42)
        print(f"Flattened input shape: {x.shape}")

    # Ensure tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32) #x: tensor float 32

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long) #y: nhãn long

    # Tạo dataset (tensor x, long y)
    dataset = Data_Loader(list(zip(x, y))) 
    # Lưu trong class DataLoader
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle) 
    # In thông báo
    print(
        f"Created DataLoader with batch_size={batch_size}, shuffle={shuffle}")
    return dataloader

"""
def prepare_datasets(dataset: pd.DataFrame, config: GANIDSConfig):
    
    #Tách dữ liệu theo số lớp tối đa cho phép, train/test split từng lớp.
    #Trả về dữ liệu huấn luyện, kiểm thử, và bản sao của DataFrame train/test.
    
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
"""

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


class GeneratorNet(nn.Module): # input: z (N, 5)
    def __init__(self, output_size): # output_size = 42/ 45
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, output_size),
            nn.Tanh()
        ) # fake_data = generator(z)  # fake_data shape: (N(size), output_size)


    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module): # output: 0/ 1
    def __init__(self, input_size): # input : vector 42/ 45
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        ) # real_or_fake = discriminator(data)  # data shape: (batch_size, input_size)

    def forward(self, x):
        return self.model(x)


def get_gradients(model):
    """Trả về dict gradients của model hiện tại."""
    return {n: p.grad.clone().cpu() if p.grad is not None else torch.zeros_like(p.data).cpu() for n, p in model.named_parameters()}


def train_discriminator(discriminator, optimizer, real_data, fake_data, y_real, loss_fn, penalty_coef):
    N = real_data.size(0) # Lấy batch_size
    optimizer.zero_grad() # Xóa grad cũ

    # Dự đoán đúng trên dl thật
    prediction_real = discriminator(real_data) #Lấy output: 1/ 0
    loss_real = loss_fn(prediction_real.squeeze(), true_target(y_real).squeeze())
    # Dự đoán đúng trên dl giả
    prediction_fake = discriminator(fake_data.detach())
    loss_fake = loss_fn(prediction_fake, zeros_target(N))
    # Hệ số phạt
    penalty = penalty_coef * torch.mean((prediction_fake - 0.5) ** 2)
    # Tổng loss
    total_loss = loss_real + loss_fake + penalty
    # Tính gradient
    total_loss.backward()
    grads = get_gradients(discriminator)
    # Cập nhật weight
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
    N = fake_data.size(0) # Lấy batch_size
    optimizer.zero_grad() # xóa grad cũ
    
    #Dự đóan sai của D_l trên fake_data
    prediction = discriminator(fake_data) 
    loss = loss_fn(prediction, ones_target(N))
    # Hệ số phạt
    penalty = penalty_coef * torch.mean((prediction - 1.0) ** 2)
    total_loss = loss + penalty
    # Tính grads
    total_loss.backward()
    grads = get_gradients(generator)
    #Cập nhật weight
    optimizer.step()
    return {
        'error': loss.item(),
        'total_error': total_loss.item(),
        'gradients': grads
    }


def run_gan_training(config, x_data, y_data, target_class, num_samples, save_log=True):
    data_loader = create_dataloader(x_data, (y_data == target_class).astype(int), config.batch_size) #target class: lớp yếu
    input_size = 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45 #set số đặc trưng

    #Chạy GAN
    D = DiscriminatorNet(input_size=input_size)
    G = GeneratorNet(output_size=input_size)
    #Nếu có trọng số trước đó, nạp lại để huấn luyện
    if config.disc_params:
        D.load_state_dict(config.disc_params)
    if config.gen_params:
        G.load_state_dict(config.gen_params)

    # Cập nhật tham số = Adam
    d_opt = optim.Adam(D.parameters(), lr=config.lr_disc)
    g_opt = optim.Adam(G.parameters(), lr=config.lr_gen)
    # Hàm mất mát là Binary Cross-Entropy (đúng/sai).
    loss_fn = nn.BCELoss()
    # Thông tin huấn luyện 
    history = {'disc_losses': [], 'gen_losses': [], 'disc_feedback': [], 'gen_feedback': []}
    last_gen_grads = None
    last_disc_grads = None
    last_gen_error = None
    last_disc_error = None
    # Lặp huấn luyện (local_epoch)
    for epoch in range(config.gan_epochs):
        disc_epoch, gen_epoch = [], []
        #Dữ liệu thật
        for real_batch, y_real in data_loader:  
            N = real_batch.size(0) # batch_size
            real_data = real_batch # data

            # Sinh dữ liệu giả
            fake_data = G(noise(N)).detach()
            y_real = y_real.numpy() #Mảng nhãn 2D

            # Train Discriminator
            disc_fb = train_discriminator(D, d_opt, real_data, fake_data, y_real, loss_fn, config.penalty_coef)
            # Train Generator
            gen_fb = train_generator(D, G, g_opt, G(noise(N)), loss_fn, config.penalty_coef)
            
            # Ghi feedback của G_l, D_l trong lần lặp này vào mảng feedback chung
            disc_epoch.append(disc_fb)
            gen_epoch.append(gen_fb)

            #Lưu feedback cuối cùng 
            last_disc_grads = disc_fb['gradients']
            last_disc_error = disc_fb['total_error']
            last_gen_grads = gen_fb['gradients']
            last_gen_error = gen_fb['total_error']
        
        # Tính tổng lỗi của D_l, G_l 
        history['disc_losses'].append(sum(f['total_error'] for f in disc_epoch))
        history['gen_losses'].append(sum(f['total_error'] for f in gen_epoch))
        history['disc_feedback'].append(disc_epoch)
        history['gen_feedback'].append(gen_epoch)

        # In log mỗi 5 vòng
        if epoch % 5 == 0:
            print(f"Class {target_class} - Epoch {epoch} | D Loss: {history['disc_losses'][-1]:.4f} | G Loss: {history['gen_losses'][-1]:.4f}")
    
    #--------------------------------------------
    # Sinh dữ liệu giả (số lượng num_samples) và (lớp yếu target class) 
    x_syn = G(noise(num_samples)).detach().numpy()
    y_syn = np.ones(num_samples) * target_class

    # Save_log
    if save_log:
        with open(f"gan_results_class_{target_class}.txt", "w") as f:
            for epoch in range(config.gan_epochs):
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
        'history': history,
        'gen_grads': last_gen_grads,
        'disc_grads': last_disc_grads,
        'gen_error': last_gen_error,
        'disc_error': last_disc_error
    }


def train_ids_model(x_train, y_train, x_test, y_test, config):
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Chuyển mảng NumPy -> tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Tạo data loaders
    train_loader = create_dataloader(
        x_train_tensor, y_train_tensor, config.batch_size)
    test_loader = create_dataloader(
        x_test_tensor, y_test_tensor, config.batch_size)

    num_classes = len(np.unique(y_train))  # Số class 

    # Set input_size 
    input_size = 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45

    #Gọi model CNN  --------------------------------------------
    # Đổi đầu vào thành x_reshape 4D cho CNN 2D
    model = CNNClassifier(input_size=input_size, num_classes=num_classes)
    print(
        f"Initialized CNNClassifier with input_size={input_size}, num_classes={num_classes}")

    optimizer = optim.Adam(model.parameters(), lr=0.001) # cập nhật trọng số
    criterion = nn.CrossEntropyLoss() #hàm loss (ko cần sửa)
    
    # Lưu kết quả huấn luyện
    history = {'train_losses': [], 'test_losses': [], 'train_accuracies': [
    ], 'test_accuracies': [], 'f1_scores': [], 'reports': []}
    
    # Huấn luyện IDS (gloal_epoch) ------------------------
    for epoch in range(config.ids_epochs):
        model.train() # huấn luyện
        correct, total, loss_val = 0, 0, 0 #số mẫu dự đoán đúng, tổng số mẫu, tổng loss
        
        #Lặp qua từng batch train_loader (data train)
        # x: tensor (batch_size, input_size), target: nhãn
        for batch_idx, (x, target) in enumerate(train_loader):
            print( 
                f"Epoch {epoch+1}, Batch {batch_idx+1}: x.shape={x.shape}, target.shape={target.shape}")
            
            optimizer.zero_grad() # Xóa grad
            out = model(x) #output của batch data_train

            # Tính loss và gradient
            loss = criterion(out, target) 
            loss.backward()
            optimizer.step()
            
            # Dự đoán
            pred = out.argmax(dim=1) # Tính nhãn dự đoán của mô hình
            correct += (pred == target).sum().item() # correct ++ nếu dự đoán đúng
            total += x.size(0) # số lượng mẫu
            loss_val = loss_val * 0.9 + loss.item() * 0.1 #trung bình loss
        
        train_acc = correct / total # độ chính xác
        history['train_losses'].append(loss_val)
        history['train_accuracies'].append(train_acc)
        print(
            f"Epoch {epoch+1} - Train Loss: {loss_val:.4f}, Train Acc: {train_acc:.4f}")

        # Đánh giá model ------------------------------------------
        model.eval()
        correct, total, loss_val = 0, 0, 0
        all_preds = [] # lưu dự đoán
        all_targets = [] # lưu kết quả đúng 

        with torch.no_grad(): # tắt cập nhật trọng số 
            for batch_idx, (x, target) in enumerate(test_loader): # Lặp qua data test
                print(
                    f"Test Batch {batch_idx+1}: x.shape={x.shape}, target.shape={target.shape}")
                
                out = model(x) # Train model
                loss = criterion(out, target) # Tính loss
                # Dự đoán (tương tự phần huấn luyện IDS)
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
        
        # Tính độ chính xác
        test_acc = correct / total
        history['test_losses'].append(loss_val)
        history['test_accuracies'].append(test_acc)
        print(
            f"Epoch {epoch+1} - Test Loss: {loss_val:.4f}, Test Acc: {test_acc:.4f}")

    # Chuyển nhãn dự đoán, kết quả đúng từ list -> numpy array
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Tính phân loại cho toàn bộ dữ liệu (class và các chỉ số)
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


def receive_config_from_server(client_socket):
    config_dict = receive_data(client_socket)
    config = GANIDSConfig()
    config.set_training_params(config_dict)
    return config


def train_gan_local(config, x_train, y_train):
    input_size = 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45
    generator = GeneratorNet(output_size=input_size)
    discriminator = DiscriminatorNet(input_size=input_size)
    # Có thể load weights từ server nếu cần
    # ... train GAN như run_gan_training ...
    # Ở đây chỉ khởi tạo và trả về state_dict để demo
    return generator, discriminator


def run_gan_ids_pipeline(verbose=True, client_id=0, host='127.0.0.1', port=8888):
    if verbose:
        print("=== Starting FEDGAN-IDS Client Pipeline ===")
        print(f"Client ID: {client_id}")
    try:
        # 0. Kết nối với server 
        client_socket = connect_to_server(host, port) 
        if verbose:
            print(f"Connected to server at {host}:{port}")
        
        # 1. Nhận config từ server
        if verbose:
            print("1. Receiving config from server...")
        config = receive_config_from_server(client_socket)
        if verbose:
            print(f"Received config: {vars(config)}")
       
       # 2. Load và preprocess data
        if verbose:
            print("2. Loading and preprocessing data...")
            
            # x: 4D (đã reshape), y: nhãn dạng số, _df: 2D, có nhãn (data đã xử lí)
        x_train, x_test, y_train, y_test, train_df, test_df = load_and_preprocess_data(config)

        # 3. Train IDS trước khi train GAN
        if verbose:
            print("3. Training initial IDS model (before GAN update)...")
        initial_results = train_ids_model(x_train, y_train, x_test, y_test, config)
        f1_initial = initial_results['history']['f1_scores'][-1] if initial_results['history']['f1_scores'] else f1_score(initial_results['true_labels'], initial_results['predictions'], average=None)
        print("Initial IDS F1-score per class:", f1_initial)
        
        # 4. Train local GAN (Generator/Discriminator) #input: thêm xử lí lớp yếu
        if verbose:
            print("4. Training local GAN...")
        generator, discriminator = train_gan_local(config, x_train, y_train)
        
        # 5. Gửi local GAN weights lên server, thiếu tổng lỗi
        if verbose:
            print("5. Sending local GAN weights to server...")
        send_data(client_socket, {
            'client_id': client_id,
            'gen_state': generator.state_dict(),
            'disc_state': discriminator.state_dict(),
            'input_size': 42 if config.dataset_type in ['KDD99', 'NSL-KDD'] else 45
        })
        
        # 6. Nhận lại weights mới từ server (nhận thêm lỗi)
        if verbose:
            print("6. Receiving updated GAN weights from server...")
        response = receive_data(client_socket)
        new_gen_weights = response['new_gen_weights']
        new_disc_weights = response['new_disc_weights']
        
        # 7. Cập nhật lại model local GAN
        if verbose:
            print("7. Updating local GAN with new weights...")
        generator.load_state_dict(new_gen_weights)
        discriminator.load_state_dict(new_disc_weights)
         # Phải train lại GAN
         #........   

        # 8. (Optional) Augment dữ liệu bằng GAN mới nếu muốn
        if verbose:
            print("8. Training IDS model after GAN update...")
            # Train lại IDS 
        augmented_results = train_ids_model(x_train, y_train, x_test, y_test, config)
        f1_augmented = augmented_results['history']['f1_scores'][-1] if augmented_results['history']['f1_scores'] else f1_score(augmented_results['true_labels'], augmented_results['predictions'], average=None)
        print("Augmented IDS F1-score per class:", f1_augmented)
        
        # 9. Vẽ biểu đồ F1-score trước/sau
        plot_f1_scores(f1_initial, f1_augmented, dataset_name=config.dataset_type, client_id=client_id)
        client_socket.close()
        if verbose:
            print("Pipeline completed!")
        return True
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        if 'client_socket' in locals():
            client_socket.close()
        raise

# Hàm vẽ sơ đồ
def plot_f1_scores(initial_f1, augmented_f1, dataset_name=None, client_id=None):
    min_len = min(len(initial_f1), len(augmented_f1))
    x = np.arange(min_len)
    width = 0.35

    plt.bar(x - width/2, initial_f1[:min_len], width, label='Original')
    plt.bar(x + width/2, augmented_f1[:min_len], width, label='Augmented')

    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    title = "F1 Score per Class"
    if dataset_name or client_id is not None:
        title += f" | Dataset: {dataset_name} | Client: {client_id}"
    plt.title(title)
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'f1_scores_{dataset_name}_client{client_id}.png')
    plt.show()

# Save model (bỏ cũng đc)
def save_model(model_state, filename):
    torch.save(model_state, filename)
    print(f"Model saved to {filename}")

#Set global epochs (nhận từ server)--------- 
def run_global_training(config, n_runs=5):
    avg_improvements = []
    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} ===")
        results = run_gan_ids_pipeline(
        verbose=True, client_id=0, host='127.0.0.1', port=8888)

        f1_initial = results['initial_ids']['history']['f1_scores'][-1]
        f1_augmented = results['augmented_ids']['history']['f1_scores'][-1]
        improvement = np.mean(f1_augmented) - np.mean(f1_initial)
        print(f"Run {i+1} Improvement: {improvement:.4f}")
        avg_improvements.append(improvement)
    print(
        f"\n>>> Average Improvement over {n_runs} runs: {np.mean(avg_improvements):.4f} ± {np.std(avg_improvements):.4f}")

#----------------------------
# Tương tác với server 
def connect_to_server(host='127.0.0.1', port=8888):
    """Establish connection to the server."""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket

def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)  # object data sang byte
    # Gửi 4 byte đầu tiên cho bên nhận biết tổng số byte dữ liệu sẽ được gửi
    client_socket.sendall(struct.pack('!I', len(serialized_data)))
    client_socket.sendall(serialized_data)

def receive_data(client_socket):
    # Nhận 4 byte độ dài dữ liệu sẽ được nhận
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
    return pickle.loads(data)  # deserialize

#-------------------------------
# === MAIN ===
if __name__ == '__main__':

    # Ghi log ---------
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
    
    # Bắt đầu chương trình 
    try:
        # Initialize configuration - Khởi tạo tham số cần thiết
        config = GANIDSConfig()

        config.prompt_dataset_type()  # ← Gọi hàm yêu cầu chọn bộ dữ liệu

        # Validate configuration
        validate_config(config) # Kiểm tra tính hợp lệ của dữ liệu

        # Run the pipeline with socket communication 
        # Nên chạy run_global_training
        run_gan_ids_pipeline(verbose=True, client_id=client_id, host='127.0.0.1', port=8888)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()