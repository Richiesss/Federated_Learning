# strategies/base_strategy.py

import os
import csv
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from models.unet import UNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BaseStrategy(fl.server.strategy.FedAvg):
    """FedAvgベースのカスタムストラテジー"""

    def __init__(
        self, strategy_name: str, initial_parameters: Parameters = None, **kwargs
    ):
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.strategy_name = strategy_name
        self.initial_parameters = initial_parameters
        self.current_weights: List[np.ndarray] = []
        if initial_parameters is not None:
            self.current_weights = fl.common.parameters_to_ndarrays(
                self.initial_parameters
            )
        self.csv_file_path = f"{strategy_name}_metrics.csv"
        self._initialize_csv()



    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """初期パラメータをサーバー側で初期化する"""
        if self.initial_parameters is not None:
            return self.initial_parameters
        parameters = super().initialize_parameters(client_manager)
        if parameters is not None:
            self.initial_parameters = parameters
            self.current_weights = fl.common.parameters_to_ndarrays(parameters)
            return parameters
        else:
            print("Warning: parameters is None. Using default initialization.")
            model = UNet()
            initial_parameters = fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for val in model.state_dict().values()]
            )
            self.initial_parameters = initial_parameters
            self.current_weights = fl.common.parameters_to_ndarrays(initial_parameters)
            return initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """パラメータ更新を集約し、クライアントのデータ数をプロットする"""

        # クライアントのデータ数を収集
        client_data_sizes = {}
        for client, fit_res in results:
            cid = client.cid
            num_examples = fit_res.num_examples
            client_data_sizes[cid] = num_examples

        # プロットの作成
        self.plot_client_data_distribution(server_round, client_data_sizes)

        # 既存のaggregate_fitを呼び出す
        return super().aggregate_fit(server_round, results, failures)

    def plot_client_data_distribution(self, server_round, client_data_sizes):
        """各ラウンドのクライアントのデータ分布をプロット"""
        client_ids = [cid[-4:] for cid in client_data_sizes.keys()]
        data_sizes = list(client_data_sizes.values())

        plt.figure(figsize=(10, 6))
        plt.bar(client_ids, data_sizes)
        plt.xlabel("Client ID (last 4 digits)")
        plt.ylabel("Number of Samples")
        plt.title(f"Round {server_round} Client Data Distribution")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"round_{server_round}_client_data_distribution.png")
        plt.close()


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """評価結果を集約し、平均Diceスコア、Precision、Recall、Accuracy、IoUを計算する"""
        if not results:
            return None, {}

        total_examples = sum([res.num_examples for _, res in results])
        total_dice = sum(
            [res.metrics["dice_score"] * res.num_examples for _, res in results]
        )
        total_precision = sum(
            [res.metrics["precision"] * res.num_examples for _, res in results]
        )
        total_recall = sum(
            [res.metrics["recall"] * res.num_examples for _, res in results]
        )
        total_accuracy = sum(
            [res.metrics["accuracy"] * res.num_examples for _, res in results]
        )
        total_iou = sum([res.metrics["IoU"] * res.num_examples for _, res in results])

        avg_dice_score = total_dice / total_examples
        avg_precision = total_precision / total_examples
        avg_recall = total_recall / total_examples
        avg_accuracy = total_accuracy / total_examples
        avg_iou = total_iou / total_examples

        print(
            f"Round {server_round} Metrics: Dice Score = {avg_dice_score:.4f}, Precision = {avg_precision:.4f}, Recall = {avg_recall:.4f}, Accuracy = {avg_accuracy:.4f}, IoU = {avg_iou:.4f}"
        )
        self._write_metric_to_csv(
            server_round,
            avg_dice_score,
            avg_precision,
            avg_recall,
            avg_accuracy,
            avg_iou,
        )

        return super().aggregate_evaluate(server_round, results, failures)

    def _write_metric_to_csv(
        self,
        round_num: int,
        dice_score: float,
        precision: float,
        recall: float,
        accuracy: float,
        iou: float,
    ):
        """メトリックをCSVファイルに書き込む"""
        with open(self.csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([round_num, dice_score, precision, recall, accuracy, iou])

    def _initialize_csv(self):
        """CSVファイルを初期化する"""
        if os.path.exists(self.csv_file_path):
            os.remove(self.csv_file_path)
        with open(self.csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Round", "Dice Score", "Precision", "Recall", "Accuracy", "IoU"]
            )
