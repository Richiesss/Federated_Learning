# run.py
import os
import sys
import time
import argparse
from pathlib import Path
import torch
import flwr as fl
from server import get_strategy
from client import get_client_fn
from data.dataset import initialize_datasets
from models.unet import UNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))


def main():
    """メイン関数: Flowerを使用して連合学習をシミュレーションする"""
    parser = argparse.ArgumentParser(description="Federated Learning with Flower")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["SOFA", "FA"],
        help="ストラテジーを選択: SOFA もしくは FA",
    )
    parser.add_argument("--iid", action="store_true", help="IIDデータ分布を使用する")
    parser.add_argument("--gpu", action="store_true", help="GPUを使用するかどうか")
    args = parser.parse_args()

    # 初期パラメータの取得
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )

    # ストラテジーの選択
    strategy = get_strategy(args.strategy, initial_parameters)

    num_clients = 100  # クライアント数の設定
    client_resources = {"num_cpus": 1}

    # GPUが使用可能であればクライアントごとにGPUを割り当てる
    if args.gpu and torch.cuda.is_available():
        client_resources["num_gpus"] = 1

    # データセットの初期化
    initialize_datasets()

    # クライアント関数の取得
    client_fn = get_client_fn(iid=args.iid)

    # シミュレーションの開始
    start_time = time.perf_counter()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=450),
        ray_init_args={"include_dashboard": False},
    )
    end_time = time.perf_counter()

    # 実行時間の表示
    print(f"{(end_time - start_time) / 60:.2f} min")
    print("シミュレーションが完了しました。")


if __name__ == "__main__":
    main()
