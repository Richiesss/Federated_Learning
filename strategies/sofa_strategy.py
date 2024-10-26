# strategies/sofa_strategy.py

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flwr.common import Parameters, Scalar
from strategies.base_strategy import BaseStrategy


class SOFAStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="SOFA", **kwargs)
        self.Pn: Set[Tuple[str, str]] = set()  # 類似ペアリスト
        self.T: float = 0.8  # コサイン類似度の閾値

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        sample_size, _ = self.num_fit_clients(client_manager.num_available())
        client_dict = client_manager.all()
        available_clients = set(client_dict.keys())
        removed_clients = set()
        selected_clients = []

        while len(selected_clients) < sample_size and available_clients:
            num_needed = sample_size - len(selected_clients)
            remaining_clients = available_clients - removed_clients
            if len(remaining_clients) < num_needed:
                print(
                    f"Not enough clients to sample. Needed: {num_needed}, Available: {len(remaining_clients)}"
                )
                break
            sampled_client_ids = np.random.choice(
                list(remaining_clients), size=num_needed, replace=False
            )
            sampled_clients = [client_dict[cid] for cid in sampled_client_ids]
            selected_clients.extend(sampled_clients)

            print(
                f"[Round {server_round}] Initially selected client IDs (last 4 digits):"
            )
            print(", ".join([client.cid[-4:] for client in selected_clients]) + "\n")

            # コンフリクトのチェック
            conflicts = []
            for i in range(len(selected_clients)):
                for j in range(i + 1, len(selected_clients)):
                    cid1, cid2 = (
                        selected_clients[i].cid,
                        selected_clients[j].cid,
                    )
                    pair = tuple(sorted([cid1, cid2]))
                    if pair in self.Pn:
                        conflicts.append((selected_clients[i], selected_clients[j]))

            if conflicts:
                print(f"Conflicts detected in round {server_round}:")
                for client1, client2 in conflicts:
                    client_to_remove = client1 if np.random.rand() < 0.5 else client2
                    if client_to_remove in selected_clients:
                        selected_clients.remove(client_to_remove)
                        removed_clients.add(client_to_remove.cid)
                        print(
                            f"   Removing client {client_to_remove.cid[-4:]} from selection."
                        )
                print()
            else:
                break

        if len(selected_clients) < sample_size:
            print(
                f"Warning: Could not sample enough clients. Required: {sample_size}, Selected: {len(selected_clients)}"
            )

        fit_ins = fl.common.FitIns(parameters, {})
        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """パラメータ更新を集約し、クライアント間のコサイン類似度を計算する。"""
        if not results:
            return None, {}

        client_deltas = {}
        used_client_ids = []
        for client, fit_res in results:
            cid = client.cid
            used_client_ids.append(cid[-4:])
            updated_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # 更新された重みと現在の重みとの差分を計算
            delta = [u - c for u, c in zip(updated_weights, self.current_weights)]

            # 差分を平坦化
            delta_flat = np.concatenate([d.flatten() for d in delta])
            client_deltas[cid] = delta_flat

        client_ids = list(client_deltas.keys())

        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                cid1, cid2 = client_ids[i], client_ids[j]
                delta1, delta2 = client_deltas[cid1], client_deltas[cid2]

                # コサイン類似度を計算
                cos_sim = np.dot(delta1, delta2) / (
                    np.linalg.norm(delta1) * np.linalg.norm(delta2)
                )

                # コサイン類似度を出力
                print(
                    f"Round {server_round} cosine similarity between {cid1[-4:]} and {cid2[-4:]}: {cos_sim:.4f}"
                )

                # 高いコサイン類似度のペアをリストに追加
                if cos_sim > self.T:
                    print(
                        f" - Pair {cid1[-4:]} and {cid2[-4:]} added to Pn due to high cosine similarity: {cos_sim:.4f}"
                    )
                    pair = tuple(sorted([cid1, cid2]))
                    self.Pn.add(pair)

        print(
            f"Clients used in aggregate_fit for round {server_round} (last 4 digits):"
        )
        print(", ".join(used_client_ids) + "\n")

        # Fitの集約
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None, {}

        aggregated_parameters, aggregated_metrics = aggregated_result
        if aggregated_parameters is not None:
            self.current_weights = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

        return aggregated_parameters, aggregated_metrics
