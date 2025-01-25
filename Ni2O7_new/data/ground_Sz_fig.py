import numpy as np
import matplotlib.pyplot as plt
tpds = np.arange(0, 5.2, 0.2)
dft_tpd = 1.58
hole_num = 5
pressure = 29.5
keys = ['Sz = 1/2', 'Sz = 3/2', 'Sz = 5/2']
with open(f'ground_energy_pressure={pressure}GPa.txt', 'r') as f:
    data = {'Sz = 1/2': [], 'Sz = 3/2': [], 'Sz = 5/2': []}
    for line in f:
        if 'Sz' in line:
            if '0.5' in line:
                key = 'Sz = 1/2'
            if '1.5' in line:
                key = 'Sz = 3/2'
            if '2.5' in line:
                key = 'Sz = 5/2'
            continue
        data[key].append(float(line.strip()))
assert len(data['Sz = 1/2']) == len(tpds), 'data length not match'
delta_val0 = np.zeros_like(tpds)
delta_val1 = np.array(data['Sz = 3/2']) - np.array(data['Sz = 1/2'])
delta_val2 = np.array(data['Sz = 5/2']) - np.array(data['Sz = 3/2'])
fig, ax = plt.subplots()

ax.plot(tpds, delta_val2, 'b:', label=keys[2], marker='o')
ax.plot(tpds, delta_val1, 'y--', label=keys[1], marker='^')
ax.plot(tpds, delta_val0, 'k.', label=keys[0])

ax.scatter(dft_tpd, 0, color='r', s=250, marker='*')
ax.set_xlabel('tpd', fontsize=13)
ax.set_ylabel(r'$\Delta$E (eV)', fontsize=13)
ax.legend(bbox_to_anchor=(1, 1), loc=2)
ax.set_title(f'{hole_num} holes at {pressure} GPa')
fig.subplots_adjust(right=0.78)
fig.show()
fig.savefig(f'ground_energy_pressure={pressure}GPa.pdf')
