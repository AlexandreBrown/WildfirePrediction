import numpy as np


class NoDataValuePreprocessor:
    def __init__(self, no_data_fill_value: float):
        self.no_data_fill_value = no_data_fill_value
    
    def replace_no_data_values(self, data: np.ndarray, no_data_value) -> np.ndarray:
        data[data == no_data_value] = self.no_data_fill_value
        return data
    
    def get_no_data_fill_value(self) -> float:
        return self.no_data_fill_value
