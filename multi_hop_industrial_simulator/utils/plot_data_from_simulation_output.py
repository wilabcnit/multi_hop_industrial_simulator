# Save the output to a JSON file
import os
import sys

from multi_hop_industrial_simulator.utils.plot_data import plot_curves
from multi_hop_industrial_simulator.utils.plot_data import plot_curves


save_file_name = 'S_vs_N-TB_vs_TL'
y1_data = [1.349728, 1.555568, 1.275936, 1.18872]  # TL
y2_data = [1.985968, 1.761712, 1.029952, 0.87792]  # TB, ttl 3
y3_data = [2.057904, 2.233504, 1.957488, 1.610672]  # TB, ttl 12

x = [6, 12, 18, 24]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
                                ci_data=None,
                                x_label=r'Number of UEs ($N$)',
                                y_label=r'Network Throughput ($S [Gbit/s]$)',
            legends=['TL', r'TB, TTL = 3', r'TB, TTL = 12'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['', '', '.'], width=0.2)

save_file_name = '20250428_L_vs_N-TB_vs_TL_deliverable_d4'
y1_data = [0.826525032, 1.068587899, 1.526592502, 1.71272702]  # TL
y2_data = [0.623221133, 0.9395242, 1.370973883, 1.569869841]  # TB, ttl 3
y3_data = [0.621100357, 0.930663863, 1.243826079, 1.381787266]  # TB, ttl 12

x = [6, 12, 18, 24]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
                                ci_data=None,
                                x_label=r'Number of UEs ($N$)',
                                y_label=r'Average Latency ($\overline{L} [\mu s]$)',
            legends=['TL', r'TB, TTL = 3', r'TB, TTL = 12'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['', '', '.'], width=0.2)

save_file_name = 'S_vs_P-TLvsTB'

y4_data = [2.186256, 2.667712, 2.744448, 2.192832, 1.59824] # TB W=10
y3_data = [2.233504, 2.484544, 2.13096, 1.428928, 1.09392] # TB W=5
y2_data = [1.552384, 1.909632, 1.99608, 1.929152, 1.78728]  # TL W=10
y1_data = [1.555568, 1.75584, 1.774704, 1.681216, 1.53992]  # TL W=5

x = [20, 40, 60, 80, 100]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
                                ci_data=None,
                                x_label=r'Payload ($P$ [bytes])',
                                y_label=r'Network Throughput ($S$ [Gbit/s])',
                                legends=['TL, W = 5', 'TL, W = 10', r'TB, W = 5', r'TB, W = 10'],
                                marker=None, x_ticks=None, y_ticks=None,
                                colors=['darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
                                show_grid=True,
                                save_file=save_file_name,
                                textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
                                hatches=['', '.', '', '.'], width=0.2)

save_file_name = 'L_vs_P-TLvsTB'

y4_data = [1.006405675, 1.308833202, 1.504653401, 1.668432214, 1.821442022] # TB W=10
y3_data = [0.930663863, 1.194444751, 1.351238706, 1.490966037, 1.606506144] # TB W=5
y2_data = [1.212958627, 1.498312819, 1.697954988, 1.863209878, 1.995770858]  # TL W=10
y1_data = [1.068587899, 1.30881803, 1.474684171, 1.611580149, 1.715538371]  # TL W=5


x = [20, 40, 60, 80, 100]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
                                ci_data=None,
                                x_label=r'Payload ($P$ [bytes])',
                                y_label=r'Average Latency ($\overline{L} \, [\mu s]$)',
                                legends=['TL, W = 5', 'TL, W = 10', r'TB, W = 5', r'TB, W = 10'],
                                marker=None, x_ticks=None, y_ticks=None,
                                colors=['darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
                                show_grid=True,
                                save_file=save_file_name,
                                textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
                                hatches=['', '.', '', '.'], width=0.2)

############# Performance comparison with AODV ####################
save_file_name = 'CDF_aodv_tb_TTL12'
y1_data = [0.142, 0.333, 0.553, 0.975, 1, 1]  # TB, 18 UE seed = 18
y2_data = [0.336, 0.486, 0.661, 0.722, 0.772, 0.789]  # AODV, 18 UE
y3_data = [0.006, 0.333, 0.421, 0.9062, 1, 1]  # TB, 24 UE, seed = 18
y4_data = [0.335, 0.429, 0.517, 0.587, 0.625, 0.652] # AODV, 24 UE
y5_data = [0, 0.333, 0.350, 0.667, 1, 1] # TL, 18 UE
y6_data =[0, 0.333, 0.333, 0.59, 0.9333, 1] # 24 UE, TL

x = [0.5, 1, 1.5, 2, 2.5, 3]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y5_data, y3_data, y4_data, y6_data],
                                ci_data=None,
            x_label=r'Per-UE Latency ($\overline{L}_j [\mu_s]$)',
            y_label=r'CDF',
            legends=['TB, N = 18', 'AODV, N = 18', 'TL, N = 18', r'TB, N = 24', r'AODV, N = 24', 'TL, N = 24'],
            marker=["o", "s", "^", "o", "s", "^"], x_ticks=None, y_ticks=None,
            colors=['darkgray', 'skyblue','dimgray', 'darkgray', 'skyblue','dimgray'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='line',
            hatches=None, width=None, label_values=None, linestyles=['-', '-','-', '--', '--', '--'])

save_file_name = 'Jain_Index_aodv_tb_tl'
y1_data = [0.842363365, 0.890934759, 0.715151961, 0.65971377]  # TL
y2_data = [0.942159468, 0.939413892, 0.819150899, 0.679049975]  # TB, TTL=12, seed = 18
y3_data = [0.619017119, 0.401249108, 0.339278206, 0.319454712]  # AODV, TTL=12


x = [6, 12, 18, 24]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
                                ci_data=None,
                                x_label=r'Number of UEs ($N$)',
                                y_label=r'Jain Index ($J$)',
            legends=['TL', 'TB', r'AODV'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro', 'skyblue'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['', '', ''], width=0.2, label_values='')


#################################### RL ###################################


save_file_name = '20250804_S_TB_vs_DDQN_12_and_24_UE-Q_and_W_TTL12_static'
y1_data = [2.234, 2.230, 2.129, 2.025]
y2_data = [1.610, 1.943, 2.026, 2.044]

x = ["1", "2", "3", "4"]

plot_curves(x_data=x, y_data=[y1_data, y2_data],
                                ci_data=None,
                                x_label=r'Configurations of $Q$ and $W$',
                                y_label=r'Network Throughput ($S [Gbit/s]$)',
            legends=['TB, 12 UEs','TB, 24 UEs'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['',''], width=0.15, label_values='',
            asymptote_y1=2.66,
            asymptote_y1_label="MADRL, 12 UEs",
            asymptote_y1_color="black",
            asymptote_y2=2.399,
            asymptote_y2_label="MADRL, 24 UEs",
            asymptote_y2_color="black")

save_file_name = '20250804_L_TB_vs_DDQN_12_and_24_UE-Q_and_W_TTL12_static'
y1_data = [0.93, 0.97, 1.04, 1.09]
y2_data = [1.38, 1.55, 1.69, 1.83]

x = ["1", "2", "3", "4"]

plot_curves(x_data=x, y_data=[y1_data, y2_data],
                                ci_data=None,
                                x_label=r'Configurations of $Q$ and $W$',
                                y_label=r'Average Latency ($\overline{L} [\mu s]$)',
            legends=['TB, 12 UEs','TB, 24 UEs'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['', ''], width=0.15, label_values='',
            asymptote_y1=0.99,
            asymptote_y1_label="MADRL, 12 UEs",
            asymptote_y1_color="black",
            asymptote_y2=1.56,
            asymptote_y2_label="MADRL, 24 UEs",
            asymptote_y2_color="black")

save_file_name = '20250804_J_Index_TB_vs_DDQN_12_and_24_UE-Q_and_W_TTL12_static'
y1_data = [0.939, 0.953, 0.947, 0.926]
y2_data = [0.679, 0.835, 0.848, 0.847]

x = ["1", "2", "3", "4"]

plot_curves(x_data=x, y_data=[y1_data, y2_data],
                                ci_data=None,
                                x_label=r'Configurations of $Q$ and $W$',
                                y_label='Jain Index ($J$)',
            legends=['TB, 12 UEs','TB, 24 UEs'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['darkgray', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['', ''], width=0.15, label_values='',
            asymptote_y1=0.955,
            asymptote_y1_label="MADRL, 12 UEs",
            asymptote_y1_color="black",
            asymptote_y2=0.834,
            asymptote_y2_label="MADRL, 24 UEs",
            asymptote_y2_color="black")



