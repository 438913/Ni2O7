import numpy as np
import matplotlib.pyplot as plt
holes = 4
Sz = 0
file_path = f'{holes}holes_type_Sz={Sz}'
with open(file_path, 'r') as f:
    state_type = {}
    for line in f:
        if ':' in line:
            key, value = line.split(':', 1)
            key, value = key.strip(), float(value.strip())
            if key not in state_type:
                state_type[key] = [0] * 2