# server.py

from typing import Optional
from flwr.common import Parameters
from strategies.sofa_strategy import SOFAStrategy
from strategies.custom_fedavg import CustomFedAvg


def get_strategy(strategy_name: str, initial_parameters: Parameters):
    if strategy_name == "SOFA":
        return SOFAStrategy(
            initial_parameters=initial_parameters,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=100,
        )
    elif strategy_name == "FA":
        return CustomFedAvg(
            initial_parameters=initial_parameters,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=100,
        )
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
