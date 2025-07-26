import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_path)

import time
from typing import List
from dataclasses import dataclass

def create_file_if_not_exist(file_path):
    try:
        with open(file_path, 'a') as file:
            pass
    except FileNotFoundError:
        floder_path = file_path[:file_path.rfind('/')]
        os.makedirs(floder_path, exist_ok=True)
        with open(file_path, 'w') as file:
            pass
        time.sleep(1)

@dataclass
class DataProblem:
    prefix_list: List = None
    problem_list: List = None
    answer_list: List = None
    
@dataclass
class RawData:
    seed_list: List = None
    problem_list: List = None
    answer_list: List = None
    cost_list: List = None