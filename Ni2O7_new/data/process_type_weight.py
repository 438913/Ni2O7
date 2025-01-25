import numpy as np
import matplotlib.pyplot as plt
tpds = np.arange(0.1, 3.1, 0.1)
tpds1 = np.arange(0.1, 2.3, 0.1)
tpds2 = np.arange(2.4, 3.0, 0.1)
dft_tpd = 1.38
hole_num = 5
pressure = 0
Upp = 2

length = len(tpds)
length1 = len(tpds1)
data = {}
with open(f'5hole_type_0GPa_Upp={Upp}_allSz', 'r') as f:
    i = 0
    for line in f:
        if 'tpd' in line:
            i += 1
        if '(' in line:
            key, value = line.split(':', 1)
            key, value = key.strip(), float(value.strip())
            if key not in data:
                data[key] = [0] * (i - 1) + [value]
            else:
                data[key] += [0] * (i - len(data[key]) - 1) + [value]
    for key, value in data.items():
        if len(value) < i:
            data[key] += [0] * (i - len(value))
data_simple = {key: value for key, value in data.items() if sum(value) > 1.2}

# 主图
fig, ax = plt.subplots()
line_styles = ['y-', 'b--', 'g-.', 'r:', 'c-', 'm--', 'y-.', 'b:', 'g-', 'r--', 'c-', 'm-']
marker_styles = ['o', 'v', '^', '<', '>', 'p', 's', '8', '*', 'h', 'H', 'D', 'd', '|', '_']
# i = 0
# for key, value in data_simple.items():
#     ax.plot(tpds[: length1], value[: length1], line_styles[i], label=key, marker=marker_styles[i])
#     ax.plot(tpds[length1: length1 + 1], value[length1: length1 + 1], line_styles[i], marker=marker_styles[i])
#     ax.plot(tpds[length1 + 1: length], value[length1 + 1: length], line_styles[i], marker=marker_styles[i])
#     i += 1

hole_types = ['d7_d8(-1)', 'd8_L_d8(0)', 'L_d8_d8(-1)', 'L_d8_d9_L(-1)',
              'd8_d8_L(1)', 'd7_d9_L(-1)']
for i, hole_type in enumerate(hole_types):
    ax.plot(tpds[: length1], data[hole_type][: length1], line_styles[i], label=hole_type, marker=marker_styles[i])
    ax.plot(tpds[length1: length1 + 1], data[hole_type][length1: length1 + 1], line_styles[i], marker=marker_styles[i])
    ax.plot(tpds[length1 + 1: length], data[hole_type][length1 + 1: length], line_styles[i], marker=marker_styles[i])
    # ax.plot(tpds, data[hole_type], line_styles[i], label=hole_type, marker=marker_styles[i])

ax.axvline(2.2, ymax=0.25, color='k', linestyle=':', linewidth=2)
ax.axvline(2.4, ymax=0.25, color='k', linestyle=':', linewidth=2)
ax.scatter(dft_tpd, -0.01, color='r', s=200, marker='*')
ax.set_xlabel(r'$t_{pd}$', fontsize=15)
ax.set_ylabel('Weight', fontsize=13)
ax.set_xlim([0, 3.1])
ax.set_ylim([-0.02, 1.0])
ax.legend(bbox_to_anchor=(0.7, 0.22), labelspacing=0.18, fontsize=10, frameon=False)

# 副图数据
tpds3 = [2.290, 2.292, 2.294, 2.296, 2.300, 2.302, 2.304, 2.306, 2.308, 2.310, 2.312, 2.314, 2.318,
         2.320, 2.322, 2.324]
data_insert = []
with open(f'weight_tpd_0GPa_Upp={Upp}', 'r') as f:
    for line in f:
        if 'weight' in line:
            value = line.split('=')[1]
            value = float(value.strip())
            data_insert.append(value)
dz2up = '$d_{z^2}^\u2191$'
dz2dn = '$d_{z^2}^\u2193$'
dz2up_dn = '$d_{z^2}^{\u2191\u2193}$'
dx2up = '$d_{x^2}^\u2191$'
dx2dn = '$d_{x^2}^\u2193$'
dx2up_dn = '$d_{x^2}^{\u2191\u2193}$'
px_up = '$p_x^\u2191$'
px_dn = '$p_x^\u2193$'
px_up_dn = r'$p_x^\u2193^{\u2191}$'
py_up = '$p_y^\u2191$'
py_dn = '$p_y^\u2193$'
py_up_dn = '$p_y^\u2193^{\u2191}$'
labels = [dz2dn + dx2up_dn + '{' + dz2dn + dx2dn + '}(-1)',
          '{' + dz2up + dx2up + '}' + dz2dn + dx2up_dn + '(1)',
          dz2dn + dx2up_dn + '[' + dx2up_dn + '](-1)',
          py_up + '{' + dz2dn + dx2dn + '}' + '{' + dz2dn + dx2dn + '}(-1)',
          '{' + dz2up + dx2up + '}' + '{' + dz2dn + dx2dn + '}' + py_up + '(1)',
          py_up + '{' + dz2dn + dx2dn + '}[' + dx2up_dn + '](-1)']
# 副图
line_styles1 = ['y-', 'k-.', 'y:', 'g-', 'c-.', 'g:']
marker1 = ['o', 'p', '^', '>']
ax_insert = fig.add_axes((0.2, 0.55, 0.37, 0.3), facecolor='none')
for i in range(2):
    ax_insert.plot([2.290, 2.292, 2.294, 2.296], data_insert[16 * i: 16 * i + 4],
                   line_styles1[3 * i], marker=marker1[2 * i])
    ax_insert.plot([2.300, 2.302, 2.304, 2.306, 2.308, 2.310, 2.312, 2.314],  data_insert[16 * i + 4: 16 * i + 12],
                   line_styles1[3 * i + 1], marker=marker1[2 * i + 1])
    ax_insert.plot([2.318, 2.320, 2.322, 2.324], data_insert[16 * i + 12: 16 * i + 16],
                   line_styles1[3 * i + 2], marker=marker1[2 * i])
ax_insert.set_xlim([2.29, 2.324])
ax_insert.set_ylim([0, 0.05])
ax_insert.set_xticks([2.29, 2.3, 2.31, 2.32])
ax_insert.legend(labels, bbox_to_anchor=(1, 1.17), loc=2, labelspacing=0.18, frameon=False)
ax.set_title('Upp = 2.0')
fig.show()
fig.savefig(f'5hole_type_weight_0GPa_Upp={Upp}.pdf')
