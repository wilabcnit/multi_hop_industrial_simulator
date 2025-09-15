# Save the output to a JSON file
import os
import sys

from timessim.utils.plot_data import plot_curves
from timessim.utils.plot_data import plot_curves
from timessim.utils.read_simulation_output import read_simulation_output
from scipy import stats
from scipy.stats import t

#sys.path.append(os.path.dirname(os.getcwd()))
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import matplotlib.pyplot as plt
######### FINAL PLOT FOR  CONFERENCE PAPER: ############################

# 1. SUCCESS PROBABILITY
#
# save_file_name = '20240801_pS_vs_N-mesh_vs_star_vs_benchmark'
# y5_data = [0.9996, 0.9921, 0.9760, 0.9663, 0.9579]  # Unslotted, Mesh Unicast Static
# y4_data = [0.9615, 0.9362, 0.9110, 0.8867, 0.8624]  # UnSlotted, Mesh Unicast Dynamic
# y3_data = [0.9266237972841723, 0.875664686, 0.8195, 0.7665, 0.7159]  # UnSlotted, Mesh BROADCAST Dynamic
# y2_data = [0.9172, 0.8299, 0.7731, 0.7451, 0.6961]  # UnSlotted, Mesh BROADCAST Statico
# y1_data = [0.4999, 0.4997, 0.4992, 0.4986, 0.4978]  # UnSlotted, Star Statico
#
# # DATA FOR CI COMPUTATION:
#
# confidence = 0.95
#
# ci_mualoha_static = []
# mualoha_static_values = []
#
# MUALOHA_static_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/4_UE_pMac_MUALOHA_CI.json"
# MUALOHA_static_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/6_UE_pMac_MUALOHA_CI.json"
# MUALOHA_static_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/8_UE_pMac_MUALOHA_CI.json"
# MUALOHA_static_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/10_UE_pMac_MUALOHA_CI.json"
# MUALOHA_static_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/12_UE_pMac_MUALOHA_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_static_4ue)
# mualoha_static_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_static_6ue)
# mualoha_static_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_static_8ue)
# mualoha_static_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_static_10ue)
# mualoha_static_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_static_12ue)
# mualoha_static_values.append(y_values_12ue)
#
# for y_values in mualoha_static_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_static.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic = []
# mualoha_dynamic_values = []
#
# MUALOHA_dynamic_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/additional_result/4_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/additional_result/6_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/8_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/10_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_dynamic_4ue)
# mualoha_dynamic_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_dynamic_6ue)
# mualoha_dynamic_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_dynamic_8ue)
# mualoha_dynamic_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_dynamic_10ue)
# mualoha_dynamic_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_dynamic_12ue)
# mualoha_dynamic_values.append(y_values_12ue)
#
# for y_values in mualoha_dynamic_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic_unicast = []
# mualoha_dynamic_unicast_values = []
#
# MUALOHA_dynamic_unicast_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/4_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/6_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/8_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/10_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_dynamic_unicast_4ue)
# mualoha_dynamic_unicast_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_dynamic_unicast_6ue)
# mualoha_dynamic_unicast_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_dynamic_unicast_8ue)
# mualoha_dynamic_unicast_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_dynamic_unicast_10ue)
# mualoha_dynamic_unicast_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_dynamic_unicast_12ue)
# mualoha_dynamic_unicast_values.append(y_values_12ue)
#
# for y_values in mualoha_dynamic_unicast_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_unicast.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_unicast_static = []
# mualoha_unicast_static_values = []
#
# MUALOHA_unicast_static_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/4_UE_pMac_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/6_UE_pMac_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/8_UE_pMac_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/10_UE_pMac_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/12_UE_pMac_MUALOHA_static_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_unicast_static_4ue)
# mualoha_unicast_static_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_unicast_static_6ue)
# mualoha_unicast_static_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_unicast_static_8ue)
# mualoha_unicast_static_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_unicast_static_10ue)
# mualoha_unicast_static_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_unicast_static_12ue)
# mualoha_unicast_static_values.append(y_values_12ue)
#
# for y_values in mualoha_unicast_static_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_unicast_static.append(confidence_interval)
#     print(confidence_interval)
#
# ci_saloha = []
# saloha_values = []  # TODO da fare!
#
# saloha_4ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/4_UE_pMac_SUALOHA_CI.json"
# saloha_6ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/6_UE_pMac_SUALOHA_CI.json"
# saloha_8ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/8_UE_pMac_SUALOHA_CI.json"
# saloha_10ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/10_UE_pMac_SUALOHA_CI.json"
# saloha_12ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/12_UE_pMac_SUALOHA_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(saloha_4ue)
# saloha_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(saloha_6ue)
# saloha_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(saloha_8ue)
# saloha_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(saloha_10ue)
# saloha_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(saloha_12ue)
# saloha_values.append(y_values_12ue)
#
# for y_values in saloha_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_saloha.append(confidence_interval)
#     print(confidence_interval)
#
# x = [4, 6, 8, 10, 12]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y5_data, y4_data],
#                                 ci_data=[ci_saloha, ci_mualoha_static, ci_mualoha_dynamic, ci_mualoha_unicast_static,
#                                          ci_mualoha_dynamic_unicast],
#                                 x_label=r'Number of UEs ($\mathrm{N}$)',
#                                 y_label=r'Success probability ($\mathrm{p_{\rm s}}$)',
#                                 legends=[r'UALOHA', r'TL-UALOHA, Static', r'TL-UALOHA, Dynamic',
#                                          r'TB-UALOHA, Static',
#                                          r'TB-UALOHA, Dynamic'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['whitesmoke', 'gainsboro', 'gainsboro', 'darkgray', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.9), plot_type='bar',
#                                 hatches=['', '', '-', '', '-'], width=0.15)
#
# ################# 2. JAIN INDEX
#
# save_file_name = '20240801_JainIndex_vs_N-mesh_vs_star_vs_benchmark'
#
# y5_data = [0.887899228, 0.902150979, 0.89517588, 0.893638914, 0.901523051]
# y4_data = [0.9697, 0.9707, 0.9678, 0.9595, 0.9617]  # UnSlotted, Mesh unicast Dynamic
# y3_data = [0.9788, 0.9808, 0.9816, 0.9832, 0.9821]  # UnSlotted, Mesh Broadcast Dynamic
# y2_data = [0.830733417, 0.820529109, 0.840306907, 0.86448169, 0.856053572]  # UnSlotted, Mesh Broadcast Static
# y1_data = [0.4986, 0.4991, 0.4987, 0.4987, 0.4985]  # UnSlotted, Star Static
#
# # DATA FOR CI COMPUTATION:
#
# confidence = 0.95
#
# ci_mualoha_static = []
# mualoha_static_values = []
#
# MUALOHA_static_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/4_UE_Jindex_MUALOHA_CI.json"
# MUALOHA_static_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/6_UE_Jindex_MUALOHA_CI.json"
# MUALOHA_static_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/8_UE_Jindex_MUALOHA_CI.json"
# MUALOHA_static_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/10_UE_Jindex_MUALOHA_CI.json"
# MUALOHA_static_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_static/12_UE_Jindex_MUALOHA_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_static_4ue)
# mualoha_static_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_static_6ue)
# mualoha_static_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_static_8ue)
# mualoha_static_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_static_10ue)
# mualoha_static_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_static_12ue)
# mualoha_static_values.append(y_values_12ue)
#
# for y_values in mualoha_static_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_static.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic = []
# mualoha_dynamic_values = []
#
# MUALOHA_dynamic_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/additional_result/4_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/6_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/8_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/10_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic/12_UE_Jindex_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_dynamic_4ue)
# mualoha_dynamic_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_dynamic_6ue)
# mualoha_dynamic_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_dynamic_8ue)
# mualoha_dynamic_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_dynamic_10ue)
# mualoha_dynamic_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_dynamic_12ue)
# mualoha_dynamic_values.append(y_values_12ue)
#
# for y_values in mualoha_dynamic_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic.append(confidence_interval)
#     print(" dynamic broadcast: ", confidence_interval)
#
# ci_mualoha_dynamic_unicast = []
# mualoha_dynamic_unicast_values = []
#
# MUALOHA_dynamic_unicast_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/4_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/6_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/8_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/10_UE_Jindex_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic/12_UE_Jindex_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_dynamic_unicast_4ue)
# mualoha_dynamic_unicast_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_dynamic_unicast_6ue)
# mualoha_dynamic_unicast_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_dynamic_unicast_8ue)
# mualoha_dynamic_unicast_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_dynamic_unicast_10ue)
# mualoha_dynamic_unicast_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_dynamic_unicast_12ue)
# mualoha_dynamic_unicast_values.append(y_values_12ue)
#
# for y_values in mualoha_dynamic_unicast_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_unicast.append(confidence_interval)
#     print("dynamic unicast: ", confidence_interval)
#
# ci_mualoha_unicast_static = []
# mualoha_unicast_static_values = []
#
# MUALOHA_unicast_static_4ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/4_UE_Jindex_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_6ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/6_UE_Jindex_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_8ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/8_UE_Jindex_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_10ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/10_UE_Jindex_MUALOHA_static_CI.json"
# MUALOHA_unicast_static_12ue = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_static/12_UE_Jindex_MUALOHA_static_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(MUALOHA_unicast_static_4ue)
# mualoha_unicast_static_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(MUALOHA_unicast_static_6ue)
# mualoha_unicast_static_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(MUALOHA_unicast_static_8ue)
# mualoha_unicast_static_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(MUALOHA_unicast_static_10ue)
# mualoha_unicast_static_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(MUALOHA_unicast_static_12ue)
# mualoha_unicast_static_values.append(y_values_12ue)
#
# for y_values in mualoha_unicast_static_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_unicast_static.append(confidence_interval)
#     print("Unicast Static: ", confidence_interval)
#
# ci_saloha = []
# saloha_values = []
#
# saloha_4ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/4_UE_Jindex_SUALOHA_CI.json"
# saloha_6ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/6_UE_Jindex_SUALOHA_CI.json"
# saloha_8ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/8_UE_Jindex_SUALOHA_CI.json"
# saloha_10ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/10_UE_Jindex_SUALOHA_CI.json"
# saloha_12ue = "./timessim/results/20240902_Paper_WCNC/ci_ualoha/12_UE_Jindex_SUALOHA_CI.json"
#
# x1_values, y_values_4ue = read_simulation_output(saloha_4ue)
# saloha_values.append(y_values_4ue)
# x2_values, y_values_6ue = read_simulation_output(saloha_6ue)
# saloha_values.append(y_values_6ue)
# x3_values, y_values_8ue = read_simulation_output(saloha_8ue)
# saloha_values.append(y_values_8ue)
# x4_values, y_values_10ue = read_simulation_output(saloha_10ue)
# saloha_values.append(y_values_10ue)
# x5_values, y_values_12ue = read_simulation_output(saloha_12ue)
# saloha_values.append(y_values_12ue)
#
# for y_values in saloha_values:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_saloha.append(confidence_interval)
#     print(confidence_interval)
#
# x = [4, 6, 8, 10, 12]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y5_data, y4_data],
#                                 ci_data=[ci_saloha, ci_mualoha_static, ci_mualoha_dynamic, ci_mualoha_unicast_static,
#                                          ci_mualoha_dynamic_unicast],
#                                 x_label=r'Number of UEs ($\mathrm{N}$)',
#                                 y_label=r'Jain Index ($J$)',
#                                 legends=[r'UALOHA', r'TL-UALOHA, Static', r'TL-UALOHA, Dynamic',
#                                          r'TB-UALOHA, Static',
#                                          r'TB-UALOHA, Dynamic'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['whitesmoke', 'gainsboro', 'gainsboro', 'darkgray', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.9), plot_type='bar',
#                                 hatches=['', '', '-', '', '-'], width=0.15)
#
# ########### GRAFICO 3
#
# save_file_name = '20240806_ps_vs_P-mesh_dynamic_vs_benchmark'
#
# y4_data = [0.7109, 0.6263, 0.5303, 0.4270, 0.3065]  # UnSlotted, Mesh unicast DYNAMIC Sì canale, Q 8
# y3_data = [0.3655, 0.1371, 0.0794, 0.0534, 0.0346]  # UnSlotted, Mesh broadcast DYNAMIC Sì canale, Q 8
# y2_data = [0.8669, 0.7649, 0.6506, 0.5243, 0.3955]  # UnSlotted, Mesh unicast DYNAMIC, Q 4
# y1_data = [0.7159, 0.5052, 0.3277, 0.2143, 0.1436]  # UnSlotted, Mesh broadcast DYNAMIC, Q 4
#
# # DATA FOR CI COMPUTATION:
#
# confidence = 0.85
#
# ci_mualoha_dynamic_Q4 = []
# mualoha_dynamic_values_Q4 = []
#
# MUALOHA_dynamic_20B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-20byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_40B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-40byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_60B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-60byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_80B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-80byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_100B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-100byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_20B = read_simulation_output(MUALOHA_dynamic_20B_Q4)
# mualoha_dynamic_values_Q4.append(y_values_20B)
# x2_values, y_values_40B = read_simulation_output(MUALOHA_dynamic_40B_Q4)
# mualoha_dynamic_values_Q4.append(y_values_40B)
# x3_values, y_values_60B = read_simulation_output(MUALOHA_dynamic_60B_Q4)
# mualoha_dynamic_values_Q4.append(y_values_60B)
# x4_values, y_values_80B = read_simulation_output(MUALOHA_dynamic_80B_Q4)
# mualoha_dynamic_values_Q4.append(y_values_80B)
# x5_values, y_values_100B = read_simulation_output(MUALOHA_dynamic_100B_Q4)
# mualoha_dynamic_values_Q4.append(y_values_100B)
#
# for y_values in mualoha_dynamic_values_Q4:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_Q4.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic_unicast_Q4 = []
# mualoha_dynamic_values_unicast_Q4 = []
#
# MUALOHA_dynamic_unicast_20B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-20byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_40B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-40byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_60B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-60byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_80B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-80byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_100B_Q4 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-100byte-Q4/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_unicast_20B = read_simulation_output(MUALOHA_dynamic_unicast_20B_Q4)
# mualoha_dynamic_values_unicast_Q4.append(y_values_unicast_20B)
# x2_values, y_values_unicast_40B = read_simulation_output(MUALOHA_dynamic_unicast_40B_Q4)
# mualoha_dynamic_values_unicast_Q4.append(y_values_unicast_40B)
# x3_values, y_values_unicast_60B = read_simulation_output(MUALOHA_dynamic_unicast_60B_Q4)
# mualoha_dynamic_values_unicast_Q4.append(y_values_unicast_60B)
# x4_values, y_values_unicast_80B = read_simulation_output(MUALOHA_dynamic_unicast_80B_Q4)
# mualoha_dynamic_values_unicast_Q4.append(y_values_unicast_80B)
# x5_values, y_values_unicast_100B = read_simulation_output(MUALOHA_dynamic_unicast_100B_Q4)
# mualoha_dynamic_values_unicast_Q4.append(y_values_unicast_100B)
#
# for y_values in mualoha_dynamic_values_unicast_Q4:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_unicast_Q4.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic_Q8 = []
# mualoha_dynamic_values_Q8 = []
#
# MUALOHA_dynamic_20B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-20byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_40B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-40byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_60B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-60byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_80B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-80byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_100B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tl_ualoha_dynamic_12UE/12UE-100byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_20B = read_simulation_output(MUALOHA_dynamic_20B_Q8)
# mualoha_dynamic_values_Q8.append(y_values_20B)
# x2_values, y_values_40B = read_simulation_output(MUALOHA_dynamic_40B_Q8)
# mualoha_dynamic_values_Q8.append(y_values_40B)
# x3_values, y_values_60B = read_simulation_output(MUALOHA_dynamic_60B_Q8)
# mualoha_dynamic_values_Q8.append(y_values_60B)
# x4_values, y_values_80B = read_simulation_output(MUALOHA_dynamic_80B_Q8)
# mualoha_dynamic_values_Q8.append(y_values_80B)
# x5_values, y_values_100B = read_simulation_output(MUALOHA_dynamic_100B_Q8)
# mualoha_dynamic_values_Q8.append(y_values_100B)
#
# for y_values in mualoha_dynamic_values_Q8:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_Q8.append(confidence_interval)
#     print(confidence_interval)
#
# ci_mualoha_dynamic_unicast_Q8 = []
# mualoha_dynamic_values_unicast_Q8 = []
#
# MUALOHA_dynamic_unicast_20B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic_12UE/12UE-20byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_40B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic_12UE/12UE-40byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_60B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic_12UE/12UE-60byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_80B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic_12UE/12UE-80byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
# MUALOHA_dynamic_unicast_100B_Q8 = "./timessim/results/20240902_Paper_WCNC/ci_tb_ualoha_dynamic_12UE/12UE-100byte-Q8/12_UE_pMac_MUALOHA_dynamic_CI.json"
#
# x1_values, y_values_unicast_20B = read_simulation_output(MUALOHA_dynamic_unicast_20B_Q8)
# mualoha_dynamic_values_unicast_Q8.append(y_values_unicast_20B)
# x2_values, y_values_unicast_40B = read_simulation_output(MUALOHA_dynamic_unicast_40B_Q8)
# mualoha_dynamic_values_unicast_Q8.append(y_values_unicast_40B)
# x3_values, y_values_unicast_60B = read_simulation_output(MUALOHA_dynamic_unicast_60B_Q8)
# mualoha_dynamic_values_unicast_Q8.append(y_values_unicast_60B)
# x4_values, y_values_unicast_80B = read_simulation_output(MUALOHA_dynamic_unicast_80B_Q8)
# mualoha_dynamic_values_unicast_Q8.append(y_values_unicast_80B)
# x5_values, y_values_unicast_100B = read_simulation_output(MUALOHA_dynamic_unicast_100B_Q8)
# mualoha_dynamic_values_unicast_Q8.append(y_values_unicast_100B)
#
# for y_values in mualoha_dynamic_values_unicast_Q8:
#     mean_data = np.mean(y_values)
#     var_data = np.var(y_values)
#     std_data = np.std(y_values)
#
#     t_crit = np.abs(t.ppf((1 - confidence) / 2, len(y_values) - 1))
#     lower_bound = mean_data - std_data * t_crit / np.sqrt(len(y_values))
#     upper_bound = mean_data + std_data * t_crit / np.sqrt(len(y_values))
#     confidence_interval = upper_bound - lower_bound
#     ci_mualoha_dynamic_unicast_Q8.append(confidence_interval)
#     print(confidence_interval)
#
# x = [20, 40, 60, 80, 100]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=[ci_mualoha_dynamic_Q4, ci_mualoha_dynamic_unicast_Q4, ci_mualoha_dynamic_Q8,
#                                          ci_mualoha_dynamic_unicast_Q8],
#                                 x_label=r'Payload ($\mathrm{P}$) [bytes]',
#                                 y_label=r'Success probability ($\mathrm{p_{\rm s}}$)',
#                                 legends=[r'TL-UALOHA, Dynamic - Q = 4', r'TB-UALOHA, Dynamic - Q = 4',
#                                          r'TL-UALOHA, Dynamic - Q = 8', r'TB-UALOHA, Dynamic - Q = 8'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['-', '-', '.', '.'], width=0.2)


################### F2F Meeting Pamplona ########################

########### GRAFICO 3

# save_file_name = '20250123_L_vs_N-TLvsTB'
#
# y3_data = [0.15984040159703788, 0.22972470227833778, 0.3838266754162843, 0.4981472309815106] # MADRL
# y2_data = [0.13429617464400253, 0.22054236938815825, 0.35274871530450673, 0.4683350832026539]  # TB
# y1_data = [0.23660446197404686, 0.27082380543527384, 0.508810385358731, 0.5960786366249178]  # TL
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} \, [\mu s]$)',
#                                 legends=[r'TL-UALOHA', r'TB-UALOHA', r'MADRL-Based UALOHA'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'dimgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', ''], width=0.2)
#
# save_file_name = '20250123_S_vs_N-TLvsTB'
#
# y3_data = [4.2157919999999995, 5.639216, 4.206112, 2.8544] # MADRL
# y2_data = [4.335392000000001,5.6923520000000005,4.304703999999999,3.0248]  # TB
# y1_data = [2.2436480000000003,3.8488320000000003,2.6797120000000003,2.442752]  # TL
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
#                                 ci_data=None,
#                                 x_label='Number of UEs ($N$)',
#                                 y_label='Network Throughput ($S \, [Gbit/s]$)',
#                                 legends=['TL-UALOHA', r'TB-UALOHA', r'MADRL-Based UALOHA'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'dimgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', ''], width=0.2)
#
# save_file_name = '20250128_S_vs_P-TLvsTB'
#
#
# y2_data = [4.404832, 6.51836, 6.7728]  # TB
# y1_data = [2.854368, 5.001, 6.16304]  # TL
#
#
# x = [20, 50, 100]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data],
#                                 ci_data=None,
#                                 x_label='Payload ($P$ [bytes])',
#                                 y_label='Network Throughput ($S$ [Gbit/s])',
#                                 legends=['TL-UALOHA', r'TB-UALOHA'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', ''], width=0.2)
#
# save_file_name = '20250128_L_vs_P-TLvsTB'
#
#
# y2_data = [0.289282299, 0.476916983, 0.738441956]  # TB
# y1_data = [0.468545084, 0.584656332, 0.811003788]  # TL
#
#
# x = [20, 50, 100]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data],
#                                 ci_data=None,
#                                 x_label='Payload ($P$ [bytes])',
#                                 y_label='Average Latency ($\overline{L} \, [\mu s]$)',
#                                 legends=['TL-UALOHA', r'TB-UALOHA'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', ''], width=0.2)
#
#
# ######## CONFERENCE RESULTS ##################
#
# save_file_name = '20250126_S_vs_N-TLvsTB_conference'
#
# y4_data = [0.8088000000000001, 0.9149600000000001, 1.023584, 1.0478239999999999, 1.1098720000000002]
# y3_data = [1.660896, 2.197872, 2.5966880000000003, 2.7835360000000002, 3.187376]
# y2_data = [2.3296479999999997, 3.30952, 4.121600000000001, 4.512912000000001, 5.169568]  # TB Statico
# y1_data = [1.7252960000000002, 2.4381120000000003, 2.767344, 3.251856, 3.535568]  # TL Statico
#
#
# x = [4, 6, 8, 10, 12]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label='Number of UEs ($N$)',
#                                 y_label='Network Throughput ($S$ [Gbit/s])',
#                                 legends=['TL, Static', r'TB, Static', 'TL, Dynamic', r'TB, Dynamic'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', '.', '.'], width=0.2)
#
# save_file_name = '20250126_L_vs_N-TLvsTB_conference'
#
# y4_data = [0.3666984333220755, 0.4010569089773704, 0.41623400113503984, 0.43347344211592384, 0.4476808467468115]
# y3_data = [0.2482269573690329, 0.28140749464078317, 0.31197216783169353, 0.35117441436372205, 0.3675428024149075]
# y2_data = [0.1846559112487698, 0.18755857090021547, 0.20509931955775306, 0.22452426848975212, 0.23115081791630029]  # TB Statico
# y1_data = [0.21014164818579773, 0.22429682407193283, 0.26526443162172786, 0.2794583873463939, 0.3019187040866032]  # TL Statico
#
#
# x = [4, 6, 8, 10, 12]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} \, [\mu s]$)',
#                                 legends=['TL, Static', r'TB, Static', 'TL, Dynamic', r'TB, Dynamic'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', '.', '.'], width=0.2)
#
# save_file_name = '20250130_S_vs_P-TLvsTB_conference'
# ###### Dinamico ############
# # y4_data = [1.22032, 1.8763840000000003, 2.149344, 2.4658559999999996, 2.4172000000000002]
# # y3_data = [2.794944, 4.122208, 4.64688, 4.993216, 5.3768]
# # y2_data = [1.1098720000000002, 1.8168000000000002, 2.185392, 2.251456, 2.28104]  # TB
# # y1_data = [3.187376, 4.650624, 5.480208, 5.953216, 6.312880000000001]  # TL
#
# ########### Statico ##########
# y4_data = [5.56304, 8.753152, 10.408464, 10.72768, 10.744879999999998]
# y3_data = [3.23136, 4.805440000000001, 5.861424000000001, 6.487552000000001, 6.757039999999999]
# y2_data = [5.169568, 8.350336, 9.506688, 10.289792, 9.94792]  # TB Q=4
# y1_data = [3.535568, 5.444608000000001, 6.174912, 7.050752000000001, 7.352320000000001]  # TL Q=4
#
# x = [20, 40, 60, 80, 100]
#
# plot_curves(x_data=x, y_data=[y1_data, y3_data, y2_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Payload ($P$ [bytes])',
#                                 y_label=r'Network Throughput ($S$ [Gbit/s])',
#                                 legends=['TL, Q = 5', 'TL, Q = 9', r'TB, Q = 5', r'TB, Q = 9'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'gainsboro', 'darkgray', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '.', '', '.'], width=0.2)
#
# save_file_name = '20250126_L_vs_P-TLvsTB_conference'
#
# y4_data = [1.22032, 1.8763840000000003, 2.149344, 2.4658559999999996, 2.4172000000000002]
# y3_data = [0.42115825877156753, 0.5500811592817761, 4.64688, 4.993216, 5.3768]
# y2_data = [0.4476808467468115, 0.4983947413734891, 0.5502247656425491, 0.6279441537573615, 0.6952478904385562]  # TB Statico
# y1_data = [0.3675428024149075, 0.4758635243937419, 0.5715779852438017, 0.6574375704618431, 0.7293051629306664]  # TL Statico
#
#
# x = [20, 40, 60, 80, 100]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} \, [\mu s]$)',
#                                 legends=['TL, Q = 4', r'TB, Q = 4', 'TL, Q = 8', r'TB, Q = 8'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['gainsboro', 'darkgray', 'gainsboro', 'darkgray'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', '.', '.'], width=0.2)
#
#
# # MOBILITY RESULTS:
#
# # SUCCESS PROBABILITY
# save_file_name = '20250304_ps_vs_N-TB-Static-Dynamic'
# y3_data = [0.719545933, 0.535291553, 0.399670179, 0.275628986] # TB DINAMICO 1 MOVIMENTO
# y2_data = [0.84973247, 0.652006204, 0.402115103, 0.288341581]  # TB DINAMICO 3 MOVIMENTI
# y1_data = [0.954907114, 0.717282491, 0.380209711, 0.268040824]  # TB STATICO
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Success probability ($p_s$)',
#                                 legends=['TB, Static', r'TB, Dynamic with 3 movements', 'TB, Dynamic with 1 movement'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '.', '-'], width=0.2)
#
# # THROUGHPUT
# save_file_name = '20250304_S_vs_N-TB-Static-Dynamic'
# y4_data = [1.75016, 1.671984, 1.229216, 1.000384] # TB DINAMICO, 1 MOVIMENTO, RL
# y3_data = [1.70408, 1.631024, 1.1332, 0.883056] # TB DINAMICO 1 MOVIMENTO
# y2_data = [1.96696, 1.825776, 1.157392, 0.956064]  # TB DINAMICO 3 MOVIMENTI
# y1_data = [1.979536, 1.807952, 1.070016, 0.906112]  # TB STATICO
#
# label1 = ["3.49", "5.77", "7.81", "10.16"]
# label2 = ["3.5", "5.84", "7.86", "10.16"]
# label3 = ["3.37", "5.78", "7.86", "10.14"]
# label4 = ["4.28", "7.57", "10.35", "13.38"]
#
# x = [6, 12, 18, 24]
#
#
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None, label_values=[label1, label2, label3, label4],
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#                                 legends=['TB, Static', r'TB, Dynamic (2)', 'TB, Dynamic (1)',
#                                          'TB with RL, Dynamic (1)'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro', 'gainsboro', 'white' ],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 2), plot_type='bar',
#                                 hatches=['', '.', '\\', '\\'], width=0.2)
#
#
#
# # LATENCY
# save_file_name = '20250304_L_vs_N-TB-Static-Dynamic'
# y3_data = [0.277387391, 0.458968894, 0.615075527, 0.682151307] # TB DINAMICO 1 MOVIMENTO
# y2_data = [0.281200831, 0.460549106, 0.612714637, 0.67914119]  # TB DINAMICO 3 MOVIMENTI
# y1_data = [0.314994835, 0.473165607, 0.624565608, 0.685944282]  # TB STATICO
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} [\mu s]$)',
#                                 legends=['TB, Static', r'TB, Dynamic with 3 movements', 'TB, Dynamic with 1 movement'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '.', '-'], width=0.2)
#
# # JAIN INDEX
# save_file_name = '20250304_Jindex_vs_N-TB-Static-Dynamic'
# y3_data = [0.742867476, 0.619420248, 0.676436015, 0.641854687] # TB DINAMICO 1 MOVIMENTO
# y2_data = [0.85306826, 0.784735616, 0.660666575, 0.594923198]  # TB DINAMICO 3 MOVIMENTI
# y1_data = [0.929328652, 0.907748413, 0.642534658, 0.515588788]  # TB STATICO
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Jain Index',
#                                 legends=['TB, Static', r'TB, Dynamic with 3 movements', 'TB, Dynamic with 1 movement'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '.', '-'], width=0.2)
#
# # THROUGHPUT G_ue = 5 dB e G_ue = 6 dB
# save_file_name = '20250304_S_vs_N-TB-5dB-6dB'
#
# y2_data = [2.1056960000000005, 1.8017120000000002, 1.0085920000000002, 0.8031200000000001]  # TB STATICO G_UE = 6dB
# y1_data = [1.979536, 1.807952, 1.070016, 0.906112]  # TB STATICO G_UE = 5 dB
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#                                 legends=['TB, $G_{ue}$ = 5 dB', r'TB, $G_{ue}$ = 6 dB'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', ''], width=0.2)
#
# # JAIN INDEX
# save_file_name = '20250304_Jindex_vs_N-TB-5dB-6dB'
# y2_data = [0.9618040988671694, 0.9724640435010283, 0.6984829085527932, 0.5450258362988105]  # TB STATICO 6 dB
# y1_data = [0.929328652, 0.907748413, 0.642534658, 0.515588788]  # TB STATICO 5 dB
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Jain Index',
#                                 legends=['TB, $G_{ue}$ = 5 dB', r'TB, $G_{ue}$ = 6 dB'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', ''], width=0.2)
#
#
# save_file_name = '20250320_Jindex_vs_N-TB_vs_TL'
# y2_data = [0.841644791, 0.884788988, 0.712237402, 0.654021697]  # TL
# y1_data = [0.929328652, 0.907748413, 0.642534658, 0.515588788]  # TB
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Jain Index',
#                                 legends=['TB', r'TL'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', ''], width=0.2)
#
# save_file_name = '20250320_ps_vs_N-TB_vs_TL'
# y1_data = [0.863559368, 0.697008272, 0.491235511, 0.381927412]  # TL
# y2_data = [0.973981749, 0.709049172, 0.365564978, 0.263241509]  # TB, ttl 3
# y3_data = [0.973732944, 0.830159288, 0.496458519, 0.293971233]  # TB, ttl 6
# y4_data = [0.977895639, 0.839797198, 0.583534465, 0.367697359]  # TB, ttl 9
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Success Probability $p_s$',
#                                 legends=['TL', r'TB, TTL = 3', r'TB, TTL = 6', r'TB, TTL = 9'],
#                                 marker=None, x_ticks=None, y_ticks=None,
#                                 colors=['darkgray', 'gainsboro', 'gainsboro', 'gainsboro'],
#                                 show_grid=True,
#                                 save_file=save_file_name,
#                                 textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#                                 hatches=['', '', '.', '-'], width=0.2)
#
# save_file_name = '20250320_S_vs_N-TB_vs_TL'
# y1_data = [1.349728, 1.549856, 1.275136, 1.188832]  # TL
# y2_data = [2.057328, 1.758176, 1.018848, 0.879456]  # TB, ttl 3
# y3_data = [2.058133333, 2.158506667, 1.472853333, 1.040426667]  # TB, ttl 6
# y4_data = [2.072106667, 2.189333333, 1.825066667, 1.345706667]  # TB, ttl 9
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#             legends=['TL', r'TB, TTL = 3', r'TB, TTL = 6', r'TB, TTL = 9'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'gainsboro', 'gainsboro', 'gainsboro'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '', '.', '-'], width=0.2)
#
# save_file_name = '20250320_L_vs_N-TB_vs_TL'
# y1_data = [0.389, 0.544439324, 0.698995527, 0.74918055]  # TL
# y2_data = [0.311699997, 0.481292156, 0.630480671, 0.692174543]  # TB, ttl 3
# y3_data = [0.310666128, 0.474433504, 0.60170739, 0.67863803]  # TB, ttl 6
# y4_data = [0.309951054, 0.474345579, 0.594409598, 0.645579132]  # TB, ttl 9
#
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} [\mu s]$)',
#             legends=['TL', r'TB, TTL = 3', r'TB, TTL = 6', r'TB, TTL = 9'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'gainsboro', 'gainsboro', 'gainsboro'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '', '.', '-'], width=0.2)
#
# ########### DELIVERABLE D4.4 TIMES #######################
save_file_name = '20250428_S_vs_N-TB_vs_TL_deliverable_d4'
y1_data = [1.349728, 1.555568, 1.275936, 1.18872]  # TL
y2_data = [1.985968, 1.761712, 1.029952, 0.87792]  # TB, ttl 3
# y3_data = [2.049744, 2.227536, 1.93008, 1.604304]  # TB, ttl 12, seed = 1
y3_data = [2.057904, 2.233504, 1.957488, 1.610672]  # TB, ttl 12, seed = 18

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
# y3_data = [0.622284493, 0.935839497, 1.249852392, 1.385355591]  # TB, ttl 12, seed = 1

y3_data = [0.621100357, 0.930663863, 1.243826079, 1.381787266]  # TB, ttl 12, seed = 18

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

save_file_name = '20250428_S_vs_P-TLvsTB_deliverable_d4'

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

save_file_name = '20250428_L_vs_P-TLvsTB_deliverable_d4'

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
#
# save_file_name = '20250508_S_vs_N-TB_dynamic_d4'
# y1_data = [2.049744, 2.227536, 1.93008, 1.604304]  # TB static
# y2_data = [1.840736, 1.993616, 1.945984, 1.59088]  # TB, dynamic 2 mov
# y3_data = [1.715536, 1.856688, 2.002128, 1.634464]  # TB, dynamic 1 mov
# y4_data = [1.748, 1.886, 2.04, 1.62] # TB with RL, 1 mov
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#             legends=['TB, Static', r'TB, Dynamic (2)', r'TB, Dynamic (1)', r'TB with MADRL, Dynamic (1)'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'gainsboro', 'gainsboro', 'white'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '', '.', '.'], width=0.2)

# save_file_name = '20250508_L_vs_N-TB_dynamic_d4'
# y1_data = [0.622284493, 0.935839497, 1.249852392, 1.385355591]  # TB, Static
# y2_data = [0.560816834, 0.889126733, 1.231327312, 1.35854891]  # TB, dynamic 2 mov
# y3_data = [0.517733999, 0.857457946, 1.206607476, 1.318066403]  # TB, dynamic 1 mov
# y4_data = [0.502, 0.846, 1.187, 1.31] # TB with RL, 1 mov
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} [\mu s]$)',
#             legends=['TB, Static', r'TB, Dynamic (2)', r'TB, Dynamic (1)', r'TB with MADRL, Dynamic (1)'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'gainsboro', 'gainsboro', 'white'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '', '.', '.'], width=0.2)



#
# save_file_name = '20250530_S_vs_N-TB_TTL3_dynamic_d4'
# y1_data = [1.294064, 1.289824, 0.947, 0.85112]  # TB TTL = 3, 100 mov
# y2_data = [1.304, 1.298, 0.984, 0.866]  # RL TTL = 3, 100 mov
# y3_data = [0.595744, 0.72264, 0.664896, 0.7004]  # TB TTL = 3, 400 mov
# y4_data = [0.652, 0.901, 0.899, 0.897] # RL TTL = 3, 400 mov
# y5_data = [1.985968, 1.761712, 1.029952, 0.87792]  # TB, ttl 3
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y5_data, y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#             legends=['TB, Static', 'TB, m = 100', r'MADRL TB, m = 100', r'TB, m = 400', r'MADRL, m = 400'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['white', 'darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['-', '', '.', '', '.'], width=0.15, label_values='')
#
# save_file_name = '20250530_S_vs_N-TB_TTL12_dynamic_d4'
# y1_data = [1.489968, 1.60512, 1.23, 1.065248]  # TB TTL = 12, 25 mov
# y2_data = [1.51, 1.628, 1.241, 1.074]  # RL TTL = 12, 25 mov
# y3_data = [0.59632, 0.658672, 0.542, 0.508992]  # TB TTL = 12, 100 mov
# y4_data = [0.645, 0.781, 0.66, 0.66] # RL TTL = 12, 100 mov
# y5_data = [2.049744, 2.227536, 1.93008, 1.604304]  # TB, ttl 12
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y5_data, y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Network Throughput ($S [Gbit/s]$)',
#             legends=['TB, Static', 'TB, m = 25', r'MADRL TB, m = 25', r'TB, m = 100', r'MADRL, m = 100'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['white', 'darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['-', '', '.', '', '.'], width=0.15, label_values='')

# save_file_name = '20250530_L_vs_N-TB_dynamic_d4'
# y1_data = [0.622284493, 0.935839497, 1.249852392, 1.385355591]  # TB, Static
# y2_data = [0.560816834, 0.889126733, 1.231327312, 1.35854891]  # TB, dynamic 2 mov
# y3_data = [0.517733999, 0.857457946, 1.206607476, 1.318066403]  # TB, dynamic 1 mov
# y4_data = [0.502, 0.846, 1.187, 1.31] # TB with RL, 1 mov
#
# x = [6, 12, 18, 24]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Number of UEs ($N$)',
#                                 y_label=r'Average Latency ($\overline{L} [\mu s]$)',
#             legends=['TB, Static', r'TB, Dynamic (2)', r'TB, Dynamic (1)', r'TB with MADRL, Dynamic (1)'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'gainsboro', 'gainsboro', 'white'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '', '.', '.'], width=0.2)

save_file_name = '20250530_L_vs_N-TB_TTL3_dynamic_d4'
y1_data = [0.525, 0.737, 0.886, 0.956]  # TB TTL = 3, 100 mov
y2_data = [0.515, 0.743, 0.916, 0.996]  # RL TTL = 3, 100 mov
y3_data = [0.589, 0.649, 0.800, 0.901]  # TB TTL = 3, 400 mov
y4_data = [0.427, 0.531, 1.00, 1.10] # RL TTL = 3, 400 mov
y5_data = [0.623221133, 0.9395242, 1.370973883, 1.569869841]
# y2_data = [0.623221133, 0.9395242, 1.370973883, 1.569869841]  # TB, ttl 3


x = [6, 12, 18, 24]

plot_curves(x_data=x, y_data=[y5_data, y1_data, y2_data, y3_data, y4_data],
                                ci_data=None,
                                x_label=r'Number of UEs ($N$)',
                                y_label=r'Average Latency ($\overline{L} [\mu s]$)',
            legends=['TB, Static', 'TB, m = 100', r'MADRL TB, m = 100', r'TB, m = 400', r'MADRL, m = 400'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['white', 'darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['-','', '.', '', '.'], width=0.15, label_values='')

save_file_name = '20250530_L_vs_N-TB_TTL12_dynamic_d4'
y1_data = [0.510, 0.730, 0.831, 0.884]  # TB TTL = 12, 25 mov
y2_data = [0.499, 0.726, 0.843, 0.897]  # RL TTL = 12, 25 mov
y3_data = [0.414958, 0.509792, 0.598, 0.672361]  # TB TTL = 12, 100 mov
y4_data = [0.422, 0.535, 0.672, 0.773] # RL TTL = 12, 100 mov
y5_data = [0.622284493, 0.935839497, 1.249852392, 1.385355591]

x = [6, 12, 18, 24]

plot_curves(x_data=x, y_data=[y5_data, y1_data, y2_data, y3_data, y4_data],
                                ci_data=None,
                                x_label=r'Number of UEs ($N$)',
                                y_label=r'Average Latency ($\overline{L} [\mu s]$)',
            legends=['TB, Static', 'TB, m = 25', r'MADRL TB, m = 25', r'TB, m = 100', r'MADRL, m = 100'],
            marker=None, x_ticks=None, y_ticks=None,
            colors=['white', 'darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
            hatches=['-', '', '.', '', '.'], width=0.15, label_values='')

# save_file_name = '20250617_CDF_aodv_tb_TTL12'
# y1_data = [0.178, 0.333, 0.333, 0.431, 0.578, 0.761, 0.889, 1, 1, 1, 1]  # TB, 18 UE
# y2_data = [0.336, 0.347, 0.486, 0.572, 0.661, 0.708, 0.722, 0.764, 0.772, 0.783, 0.789]  # AODV, 18 UE
# y3_data = [0.006, 0.329, 0.333, 0.333, 0.415, 0.615, 0.877, 1, 1, 1, 1]  # TB, 24 UE
# y4_data = [0.335, 0.356, 0.429, 0.471, 0.517, 0.552, 0.587, 0.615, 0.625, 0.64, 0.652] # AODV, 24 UE
#
# x = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
#
# plot_curves(x_data=x, y_data=[y1_data, y2_data, y3_data, y4_data],
#                                 ci_data=None,
#                                 x_label=r'Latency Threshold ($L_{\rm th} [\mu_s$)',
#                                 y_label=r'Cumulative distribution function (CDF)',
#             legends=['TB, N = 18', 'AODV, N = 18', r'TB, N = 24', r'AODV, N = 24'],
#             marker=None, x_ticks=None, y_ticks=None,
#             colors=['darkgray', 'darkgray', 'gainsboro', 'gainsboro'],
#             show_grid=True,
#             save_file=save_file_name,
#             textbox_text=None, textbox_position=(0.4, 0.95), plot_type='bar',
#             hatches=['', '.', '', '.'], width=0.2, label_values='')

save_file_name = '20250617_CDF_aodv_tb_TTL12'
# y1_data = [0.178, 0.333, 0.584, 0.989, 1, 1]  # TB, 18 UE seed = 1
y1_data = [0.142, 0.333, 0.553, 0.975, 1, 1]  # TB, 18 UE seed = 18
y2_data = [0.336, 0.486, 0.661, 0.722, 0.772, 0.789]  # AODV, 18 UE
# y3_data = [0.006, 0.333, 0.415, 0.877, 1, 1]  # TB, 24 UE
y3_data = [0.006, 0.333, 0.421, 0.9062, 1, 1]  # TB, 24 UE, seed = 18
y4_data = [0.335, 0.429, 0.517, 0.587, 0.625, 0.652] # AODV, 24 UE
y5_data = [0, 0.333, 0.350, 0.667, 1, 1] # TL, 18 UE
y6_data =[0, 0.333, 0.333, 0.59, 0.9333, 1] # 24 UE, TL

x = [0.5, 1, 1.5, 2, 2.5, 3]

plot_curves(x_data=x, y_data=[y1_data, y2_data, y5_data, y3_data, y4_data, y6_data],
                                ci_data=None,
                                x_label=r'Latency Threshold ($L_{\rm th} [\mu_s]$)',
                                y_label=r'CDF Latency',
            legends=['TB, N = 18', 'AODV, N = 18', 'TL, N = 18', r'TB, N = 24', r'AODV, N = 24', 'TL, N = 24'],
            marker=["o", "s", "^", "o", "s", "^"], x_ticks=None, y_ticks=None,
            colors=['darkgray', 'skyblue','dimgray', 'darkgray', 'skyblue','dimgray'],
            show_grid=True,
            save_file=save_file_name,
            textbox_text=None, textbox_position=(0.4, 0.95), plot_type='line',
            hatches=None, width=None, label_values=None, linestyles=['-', '-','-', '--', '--', '--'])

save_file_name = '20250619_Jain_Index_aodv_tb_tl'
y1_data = [0.842363365, 0.890934759, 0.715151961, 0.65971377]  # TL
# y2_data = [0.940210278, 0.939031105, 0.814157684, 0.676705165]  # TB, TTL=12, seed = 1
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



