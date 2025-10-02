import matplotlib.pyplot as plt
from multi_hop_industrial_simulator.utils.read_inputs import read_inputs

inputs = read_inputs('inputs.yaml')

# Old RL parameters
broadcast_ampl_factor_no_change = str(inputs.get('rl').get('router').get('alfa_broad_no_change')) #(minimum = 0.5 to keep the reward between 1 and 0)
broadcast_ampl_factor_change = str(inputs.get('rl').get('router').get('alfa_broad_change'))
unicast_ampl_factor_no_ack = str(inputs.get('rl').get('router').get('alfa_broad_no_change')) #(minimum = 0.5 to keep the reward between 1 and 0)
unicast_ampl_factor_ack = str(inputs.get('rl').get('router').get('alfa_broad_change'))
energy_factor = str(inputs.get('rl').get('router').get('energy_factor'))

TTL = str(inputs.get('rl').get('router').get('TTL'))
n_actions = str(inputs.get('rl').get('agent').get('n_actions'))
batch_size = str(inputs.get('rl').get('agent').get('batch_size'))
discount_factor = str(inputs.get('rl').get('agent').get('discount_factor'))

n_simulations_for_training = str(inputs.get('rl').get('agent').get('n_simulations_for_training'))
n_simulations = str(inputs.get('simulation').get('n_simulations'))
number_of_tx_data_per_step = inputs.get('rl').get('agent').get('number_of_tx_data_per_step')
final_number_of_ues = inputs.get('simulation').get('final_number_of_ues')

# Function to plot the rewards obtained by the agent for a specific UE
def plot_rewards_per_n_ue(input_ue_id,input_ue_simulations_reward, input_n_ue, input_save_path):
    plt.figure(input_ue_id, figsize=(8, 4))
    plt.plot(input_ue_simulations_reward)
    plt.title(f'Agent in node: {input_ue_id}')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.grid(True)
    plt.savefig(input_save_path + "_agent_" + str(input_ue_id) + "_rewards_n_ue_" + str(input_n_ue) +
                "_final_n_ues_" + str(final_number_of_ues) +
                "_number_of_tx_data_per_step_" + str(number_of_tx_data_per_step) + "_only_W"
                ".png")

    plt.close()

# Function to plot the actions taken by the agent for a specific UE (Old RL version)
def plot_actions_per_n_ue(input_ue_id,input_ue_simulations_action,input_actions_label, input_n_ue, input_save_path):
    plt.figure(100 +input_ue_id, figsize=(8, 4))

    plt.plot( input_ue_simulations_action[0], 'r', label=input_actions_label[0])  # Red solid line
    plt.plot( input_ue_simulations_action[1], 'b', label=input_actions_label[1])  # Blue dashed line
    plt.plot( input_ue_simulations_action[2], 'g', label=input_actions_label[2])  # Green dotted line
    plt.plot( input_ue_simulations_action[3], 'y', label=input_actions_label[3])  # Yellow dash-dot line
    plt.title(f'Agent in node: {input_ue_id}')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(input_save_path +"agent_"+ str(input_ue_id) + "_actions_n_ue" + str(input_n_ue) +
                "b_no_change" + broadcast_ampl_factor_no_change +
                "b_change" + broadcast_ampl_factor_change +
                "u_no_ack" + unicast_ampl_factor_no_ack +
                "u_ack" + unicast_ampl_factor_ack +
                "en_f" + energy_factor +
                "ttl" + TTL +
                "batch" + batch_size +
                "disc_f" + discount_factor +
                "n_sim_for_train" + n_simulations_for_training +
                "n_sim" + n_simulations +
                 ".png")

    plt.close()