# strategies/custom_fedavg.py

from strategies.base_strategy import BaseStrategy


class CustomFedAvg(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="FA", **kwargs)
