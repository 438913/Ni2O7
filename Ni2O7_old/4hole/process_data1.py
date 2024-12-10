import re
import numpy as np
import matplotlib.pyplot as plt
with open('./data1/simplified_state_weight.txt', 'r') as file:
    state_type = {}
    i = 0
    pressures = []
    for line in file:
        line = line.strip()
        if 'pressure' in line:
            i += 1
            start_index = line.find('=')
            end_index = line.find('G')
            pressure = float(line[start_index + 1: end_index])
            pressures.append(pressure)
        if ':' in line:
            key, value = line.split(':', 1)
            key, value = key.strip(), float(value.strip())
            if key not in state_type:
                state_type[key] = [0] * (i - 1) + [value]
            else:
                state_type[key] += [0] * (i - 1 - len(state_type[key])) + [value]
with open('./data1/lowest_eigenvalue.txt', 'r') as file:
    vals = []
    for line in file:
        if '[' in line:
            val = re.findall(r'\d+\.?\d*', line)
            val = float(val[0])
            vals.append(val)
# fig, ax = plt.subplots()
# keys = list(state_type.keys())
# values = list(state_type.values())
# labels = []
# for i in range(len(keys)):
#     is_overlapping = False
#     for j in range(i):
#         if np.allclose(values[j], values[i], atol=0.01):
#             is_overlapping = True
#             break
#     if is_overlapping:
#         labels[-1] += f'\n{keys[i]}'
#     else:
#         labels.append(keys[i])
#         ax.plot(pressures, values[i], marker='o')

# ax.legend(labels, bbox_to_anchor=(1.0, 1), loc=2,  labelspacing=1.2, frameon=False)
# ax.set_xlabel(r'P$(GPa)$', fontsize=13)
# ax.set_ylabel('Weight', fontsize=13)
# ax.set_title('4 holes  Sz=0')
# fig.subplots_adjust(right=0.76)
# fig.show()

pressures = pressures[0: 5]
vals_Sz0 = vals[0: 5]
vals_Sz1 = vals[5: 10]
vals_Sz2 = vals[10: 15]
print(pressures)
print(vals_Sz1)
print(vals_Sz2)
vals0 = np.zeros_like(pressures)
delta_vals_1 = np.array(vals_Sz1) - np.array(vals_Sz0)
delta_vals_2 = np.array(vals_Sz2) - np.array(vals_Sz0)
fig1, ax1 = plt.subplots()
ax1.plot(pressures, delta_vals_2, 'b:', label='S=2',  marker='o')
ax1.plot(pressures, delta_vals_1, 'y--', label='S=1',  marker='^')
ax1.plot(pressures, vals0, 'k-', label='S=0')
ax1.set_xlabel(r'P$(Gpa)$', fontsize=13)
ax1.set_ylabel(r'$\Delta E(eV)$', fontsize=13)
ax1.set_ylim([0, 0.17])
ax1.legend(bbox_to_anchor=(1, 1), loc=2)
ax1.set_title('4 holes')
fig1.subplots_adjust(right=0.83)
fig1.show()
