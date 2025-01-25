import re
import numpy as np
import matplotlib.pyplot as plt
holes = 5
Sz = {4: ['0', '1', '2'], 5: ['1/2', '3/2', '5/2']}
file_path = f'{holes}holes_val.txt'
with open(file_path, 'r') as f:
    vals = []
    pressures = []
    for line in f:
        if '[' in line and 'pressure' not in line:
            val = re.findall(r'\d+\.?\d*', line)
            val = float(val[0])
            vals.append(val)
        line = line.strip()
        if 'pressure' in line:
            start_index = line.find('=')
            end_index = line.find('G')
            pressure = float(line[start_index + 1: end_index])
            pressures.append(pressure)

    pressures = pressures[0: 5]
    vals_Sz0 = vals[0: 5]
    vals_Sz1 = vals[5: 10]
    vals_Sz2 = vals[10: 15]
    vals0 = np.zeros_like(pressures)
    delta_vals_1 = np.array(vals_Sz1) - np.array(vals_Sz0)
    delta_vals_2 = np.array(vals_Sz2) - np.array(vals_Sz0)
    fig1, ax1 = plt.subplots()
    ax1.plot(pressures, delta_vals_2, 'b:', label=f'Sz={Sz[holes][2]}', marker='o')
    ax1.plot(pressures, delta_vals_1, 'y--', label=f'Sz={Sz[holes][1]}', marker='^')
    ax1.plot(pressures, vals0, 'k-', label=f'Sz={Sz[holes][0]}')
    ax1.set_xlabel(r'P$(Gpa)$', fontsize=13)
    ax1.set_ylabel(r'$\Delta E(eV)$', fontsize=13)
    ax1.set_ylim([-0.1, 5.0])
    ax1.legend(bbox_to_anchor=(1, 1), loc=2)
    ax1.set_title(f'{holes} holes')
    fig1.subplots_adjust(right=0.82)
    fig1.show()
