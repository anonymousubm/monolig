import random

from active_learning.uncertainty_calculators.base_uncertainty_calculator import (
    BaseUncertaintyCalculator,
)


class Random(BaseUncertaintyCalculator):
    def __init__(
        self,
        dataset_name,
        config_name,
        current_cycle,
        trial_number,
    ):
        super().__init__()

    def calculate(self, file, debug=False):
        return random.random()
