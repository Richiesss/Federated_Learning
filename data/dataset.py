# data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import shutil  # 追加

client_data_info = {}


class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels  # ラベルのリスト
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 二値マスクの場合

        if self.transform is not None:
            # 同じシードを使って画像とマスクに同じ変換を適用
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        label = self.labels[idx]
        return image, mask, label, mask_path


# グローバル変数としてデータセットを保持
_full_dataset = None
_train_indices = None
_test_indices = None


def initialize_datasets():
    global _full_dataset, _train_indices, _test_indices

    # データ変換（リサイズとテンソル化）
    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # ベースディレクトリ
    base_dir = r"D:/workspace/scripts/Thesis_Research/FL/kvasir-seg/output1/"

    # ラベルディレクトリ
    labels = ["label_0", "label_1", "label_2", "label_3", "label_4"]

    image_paths = []
    mask_paths = []
    labels_list = []

    for label_idx, label_name in enumerate(labels):
        images_dir = os.path.join(base_dir, label_name, "images")
        masks_dir = os.path.join(base_dir, label_name, "masks")

        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))

        assert len(images_list) == len(
            masks_list
        ), f"Number of images and masks do not match in {label_name}"

        for img_name, mask_name in zip(images_list, masks_list):
            image_paths.append(os.path.join(images_dir, img_name))
            mask_paths.append(os.path.join(masks_dir, mask_name))
            labels_list.append(label_idx)  # ラベルは0から4

    # データセットの作成
    _full_dataset = KvasirDataset(
        image_paths, mask_paths, labels_list, transform=data_transform
    )

    # インデックスをシャッフル
    indices = np.random.permutation(len(_full_dataset))

    # 訓練とテストの分割
    split_ratio = 0.8
    split_index = int(split_ratio * len(_full_dataset))
    _train_indices = indices[:split_index]
    _test_indices = indices[split_index:]


def save_client_data(client_id: int, train_subset: Subset):
    """
    各クライアントが学習に使用したマスク画像を保存する。
    """
    save_dir = os.path.join("client_data", f"client_{client_id}")
    if os.path.exists(save_dir):
        # 既にディレクトリが存在する場合はスキップ
        print(f"Data for client {client_id} already saved.")
        return
    else:
        os.makedirs(save_dir)

    # サブセット内のデータを保存
    for idx in train_subset.indices:
        mask_path = _full_dataset.mask_paths[idx]
        # マスク画像をクライアントのディレクトリにコピー
        shutil.copy(mask_path, save_dir)



def load_data(client_id: int, iid: bool):
    global _full_dataset, _train_indices, _test_indices, client_data_info

    if _full_dataset is None:
        initialize_datasets()

    num_clients = 100  # クライアント数を100に設定

    if iid:
        # IIDの場合、データを均等に分割
        client_indices_list = np.array_split(_train_indices, num_clients)
        client_id = client_id % num_clients
        client_train_indices = client_indices_list[client_id]

        train_subset = Subset(_full_dataset, client_train_indices)
        test_subset = Subset(_full_dataset, _test_indices)
        print(f"Client {client_id}: {len(train_subset)} training samples (IID)")

        # クライアントのデータを保存
        save_client_data(client_id, train_subset)

        # クライアントのデータ情報を保存
        labels_in_client = np.array(_full_dataset.labels)[client_train_indices]
        label_counts = np.bincount(labels_in_client, minlength=5)
        client_data_info[client_id] = {
            "num_samples": len(train_subset),
            "label_counts": label_counts.tolist(),
        }

    else:
        # 非IIDの場合、ラベルに基づいてデータを振り分ける
        num_labels = 5  # ラベル数（label_0 ~ label_4）
        clients_per_label = 20  # 各ラベルごとに20クライアント

        client_id = client_id % num_clients

        # クライアントのラベルIDを決定（0から4）
        label_id = (
            client_id // clients_per_label
        )  # 各20クライアントが一つのラベルに対応

        # 訓練データのラベルを取得（高速化のために直接参照）
        train_labels = np.array(_full_dataset.labels)[_train_indices]

        # 訓練データのインデックスを配列化
        train_indices_array = np.array(_train_indices)

        # 指定されたラベルのデータのみを取得
        label_indices = train_indices_array[train_labels == label_id]

        # ラベル内のデータをクライアント数で分割
        split_indices = np.array_split(label_indices, clients_per_label)
        client_label_local_id = client_id % clients_per_label
        client_train_indices = split_indices[client_label_local_id]

        train_subset = Subset(_full_dataset, client_train_indices)
        test_subset = Subset(_full_dataset, _test_indices)

        print(
            f"Client {client_id}: {len(train_subset)} training samples from label {label_id} (Non-IID)"
        )

        # クライアントのデータを保存
        save_client_data(client_id, train_subset)

        # クライアントのデータ情報を保存
        labels_in_client = np.array(_full_dataset.labels)[client_train_indices]
        label_counts = np.bincount(labels_in_client, minlength=5)
        client_data_info[client_id] = {
            "num_samples": len(train_subset),
            "label_counts": label_counts.tolist(),
        }

    return train_subset, test_subset
