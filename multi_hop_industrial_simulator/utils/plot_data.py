from cProfile import label
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from multi_hop_industrial_simulator.env.machine import Machine
from multi_hop_industrial_simulator.network.bs import BS
from multi_hop_industrial_simulator.network.ue import Ue
from multi_hop_industrial_simulator.env.distribution import Distribution

font_dict = {'label_size': 22,
             'ticks_size': 22,
             'legend_size': 20}


def plot_curves(x_data, y_data, x_label, y_label, legends,
                marker=None, x_ticks=None, y_ticks=None, colors=None, show_grid=True,
                save_file=None, textbox_text=None, textbox_position: tuple = None,
                plot_type='line', hatches=None, width: float = None, ci_data=None, label_values=None, linestyles=None,
                asymptote_y1: float = None, asymptote_y1_label: str = None, asymptote_y1_color: str = 'black',
                asymptote_y1_style: str = '--', asymptote_y2: float = None, asymptote_y2_label: str = None,
                asymptote_y2_color: str = 'black', asymptote_y2_style: str = '-.'):
    """

    Args:
      ci_data(list or numpy array, optional): Contains the data for the confidence interval (Default value = None)
      x_data(list or numpy array): Contains the data for the x axis
      y_data(list or numpy array): Contains data for the y-axis. It can be a list of list, or a 2-dimensional array,
    if more than one curve has to be plotted.
      x_label(str): Label for the x-axis.
      y_label(str): Label for the y-axis
      legends(list of str): Contains the labels for the curves, with the same order of 'y_data'. If one curve
    is passed, 'legends' has to be a list with one element.
      marker(list of char, optional, optional): Contains the markers for the different curves. (Default value = None)
      x_ticks(list, optional, optional): Contains two lists or numpy array elements, one for the tick positions, one for the
    values, for the x-axis. (Default value = None)
      y_ticks(list, optional, optional): Contains two lists or numpy array elements, one for the tick positions, one for the
    values, for the y-axis. (Default value = None)
      colors(list, optional, optional): List with colors for the different curves, with the same order of 'y_data'. (Default value = None)
      show_grid(bool, optional, optional): Show or not the grid. (Default value = True)
      save_file(str, optional, optional): Path + filename to save the plot. If None, it does not save the plot. (Default value = None)
      textbox_text(str, optional, optional): Text to be inserted in the textbox (Default value = None)
      textbox_position(str, optional): Position of the textbox
      plot_type(str, optional, optional): Type of plot to generate. Either 'line' for line plots or 'bar' for bar charts. (Default value = 'line')
      hatches(list, optional, optional): List of hatch patterns for each set of bars. (Default value = None)
      width(float, optional): Width of the bars.
      textbox_position: tuple:  (Default value = None)
      width: float:  (Default value = None)
      label_values:  (Default value = None)
      linestyles:  (Default value = None)
      asymptote_y1: float:  (Default value = None)
      asymptote_y1_label: str:  (Default value = None)
      asymptote_y1_color: str:  (Default value = 'black')
      asymptote_y1_style: str:  (Default value = '--')
      asymptote_y2: float:  (Default value = None)
      asymptote_y2_label: str:  (Default value = None)
      asymptote_y2_color: str:  (Default value = 'black')
      asymptote_y2_style: str:  (Default value = '-.')

    Returns:
        None
    
    """
    rect = None

    if plot_type not in ['line', 'bar']:
        raise ValueError("Invalid plot type. Supported types are 'line' and 'bar'.")

    if plot_type == 'line':
        if type(y_data[0]) is list:
            n_runs = len(y_data)
        else:
            n_runs = 1
            y_data = [y_data]

        if type(legends) is str:
            legends = [legends]

        plt.figure(figsize=(12, 8))
        # plt.rc('text', usetex=True)  # Enable LaTeX font for labels
        plt.xlabel(x_label, fontsize=font_dict['label_size'])
        plt.ylabel(y_label, fontsize=font_dict['label_size'])
        if x_ticks is not None:
            plt.xticks(x_ticks[0], x_ticks[1], fontsize=font_dict['ticks_size'])
        else:
            plt.xticks(fontsize=font_dict['ticks_size'])
        if y_ticks is not None:
            plt.yticks(y_ticks[0], y_ticks[1], fontsize=font_dict['ticks_size'])
        else:
            plt.yticks(fontsize=font_dict['ticks_size'])
        if show_grid:
            plt.grid('on', alpha=0.5)
        for i in range(len(y_data)):
            if marker is None:
                marker[i] = "o"
        plt.ylim(-0.1, 1.1)


        # Plot Data
        if asymptote_y1 is not None:
            plt.axhline(y=asymptote_y1, color=asymptote_y1_color, linestyle=asymptote_y1_style, linewidth=2, label=asymptote_y1_label)

        if asymptote_y2 is not None:
            plt.axhline(y=asymptote_y2, color=asymptote_y2_color, linestyle=asymptote_y2_style, linewidth=2, label=asymptote_y2_label)

        for run in range(n_runs):
            if colors is not None:
                plt.plot(x_data, y_data[run], color=colors[run], label=legends[run], marker=marker[run], linestyle=linestyles[run], markersize=10)
            else:
                plt.plot(x_data, y_data[run], label=legends[run], marker=marker[run], lynestyles=linestyles[run], markersize=10)

        if legends is not None:
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=font_dict['legend_size'])

        if textbox_position is None:
            textbox_position = (0.4, 0.95)

        if textbox_text is not None:
            plt.text(textbox_position[0], textbox_position[1], textbox_text, transform=plt.gca().transAxes,
                     fontsize=24, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    elif plot_type == 'bar':
        # plt.rc('text', usetex=True)  # Enable LaTeX font for labels
        fig, ax = plt.subplots(figsize=(18, 10))
        ind = np.arange(len(x_data))

        if type(y_data[0]) is list:
            n_groups = len(y_data)
        else:
            n_groups = 1
            y_data = [y_data]

        if type(legends) is str:
            legends = [legends]  # Convert single legend string to a list containing that string

        for i, y_values in enumerate(y_data):
            if colors is not None:
                color = colors[i]
            else:
                color = None
            if hatches is not None:
                hatch = hatches[i]
            else:
                hatch = None

            if ci_data is None:
                # Plot data
                rect = ax.bar(ind + width * i, y_values, width, color=color, edgecolor='black', label=legends[i],
                       hatch=hatch)
            else:
                # Plot data
                rect = ax.bar(ind + width * i, y_values, width, color=color, edgecolor='black', label=legends[i], hatch=hatch)

                # Plot errorbar for confidence intervals
                if n_groups == 1:
                    ax.errorbar(ind + width * i, y_values, ci_data[i], fmt='none', color='black', capsize=4,
                                elinewidth= 1, markeredgewidth=1)
                else:
                    ax.errorbar(ind + width * i, y_values, ci_data[i], fmt='none', color='black', linewidth=3.5,
                                capsize= 4, elinewidth=3, markeredgewidth=1)

            if label_values is not None:
                # ax.bar_label(rect, labels=label_values[i], color='black', padding=10, fontsize=14, weight='bold',
                #              bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", alpha=0.8))
                # plt.text(2.5, 2, 'Network Offered Traffic [Gbit/s]', fontdict=dict(fontsize=15, fontweight='bold'),
                #          bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver'))
                max_y_value = np.max([np.max(y_values) for y_values in y_data])  # Get the maximum y value
                max_y_value = max(max_y_value, asymptote_y1 if asymptote_y1 is not None else 0)  # Ensure it is at least the asymptote value
                max_y_value = max(max_y_value, asymptote_y2 if asymptote_y2 is not None else 0)  # Ensure it is at least the asymptote value
                if max_y_value < 1:
                    max_y_value = 1  # Ensure the maximum y value is at least 1
                padding = 0.1 * max_y_value  # Add 10% padding to the max value
                plt.ylim(0, max_y_value + padding)  # Set the y-axis limit

                #plt.ylim(0, 1.1)
        if asymptote_y1 is not None:
            ax.axhline(y=asymptote_y1, color=asymptote_y1_color, linestyle=asymptote_y1_style, linewidth=2, label=asymptote_y1_label)

                #plt.ylim(0, 1.1)
        if asymptote_y2 is not None:
            ax.axhline(y=asymptote_y2, color=asymptote_y2_color, linestyle=asymptote_y2_style, linewidth=2, label=asymptote_y2_label)

        ax.set_ylabel(y_label, fontsize=28)
        ax.set_xlabel(x_label, fontsize=28)

        ax.set_xticks(ind + width * (n_groups - 1) / 2)
        ax.set_xticklabels(x_data)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(alpha=0.6, axis='y')

        if legends is not None:
            ax.legend(loc="lower left", fontsize=24, mode="expand", ncol=2, bbox_to_anchor=(0, 0.935, 1, 0.1))

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))

    plt.show()


def plot_factory(factory_length: float, factory_width: float, factory_height: float,
                 machine_list: List[Machine], ue_list: List[Ue], bs: BS,
                 distribution_class: Distribution, scenario_name: str, save_file: str = None):
    """
    Plot a 3D representation of the factory layout showing machines, UEs, and the base station.

    Args:
        factory_length (float): Length of the factory [m].
        factory_width (float): Width of the factory [m].
        factory_height (float): Height of the factory [m].
        machine_list (List[Machine]): List of machine objects.
        ue_list (List[Ue]): List of UE objects.
        bs (BS): Base Station object.
        distribution_class (Distribution): Distribution instance providing layout information.
        scenario_name (str): Scenario name for plot title.
        save_file (str, optional): Path prefix for saving the figure. Defaults to None.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, -50)  # Adjust elevation and azimuthal angles

    # Set axis limit
    ax.set_xlim([0, factory_length])
    ax.set_ylim([0, factory_width])
    ax.set_zlim([0, factory_height])

    # Legend items + legend labels
    legend_items = list()
    legend_labels = list()

    # Machine plot
    for i in range(distribution_class.get_number_of_machines()):
        legend_item = ax.scatter(machine_list[i].x_center, machine_list[i].y_center, machine_list[i].z_center,
                                 color='gray', s=10)

        # Add the legend only one time
        if i == 1:
            legend_items.append(legend_item)
            legend_labels.append('Machine Center')
        machine_size = machine_list[i].get_machine_size()
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='gray')

    # UE plot
    index_rt = 0
    index_nrt = 0
    index_ct = 0

    for index, ue in enumerate(ue_list):
        if ue.traffic_type == 'traffic_rt':
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='red')
            index_rt += 1

            # Add the legend only one time
            if index_rt == 1:
                legend_items.append(legend_item)
                legend_labels.append('UE RT')

        elif ue.traffic_type == 'traffic_nrt':
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='green')
            index_nrt += 1

            # Add the legend only one time
            if index_nrt == 1:
                legend_items.append(legend_item)
                legend_labels.append('UE NRT')

        elif ue.traffic_type == 'traffic_cn':
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='c')
            index_ct += 1

            # Add the legend only one time
            if index_ct == 1:
                legend_items.append(legend_item)
                legend_labels.append('UE CN')

        elif ue.traffic_type == 'traffic_fq':
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='k')
            index_ct += 1

            # Add the legend only one time
            if index_ct == 1:
                legend_items.append(legend_item)
                legend_labels.append('UE')

    # BS plot
    legend_items.append(ax.scatter(bs.x, bs.y, bs.z, s=40, color='blue', marker="^"))
    legend_labels.append('BS')

    # Insert title
    title = 'Factory layout ' + str(scenario_name)
    plt.title(title)

    # Inser labels
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    ax.set_zlabel('Factory height [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels)

    # Get current date
    current_date = datetime.now()

    # Format the date components
    year = current_date.year
    month = current_date.month
    day = current_date.day

    # Save file
    if save_file is not None:
        final_file_name_plot = save_file + f'{year}_{month:02d}_{day:02d}_' + f'plot_scenario_{scenario_name}_3d.png'
        plt.savefig(final_file_name_plot, dpi=300)
        print('Output plot saved in {}'.format(final_file_name_plot))

    # Show the plot
    plt.show()


def write_data(val, file):
    """
    Write numeric data (e.g., SNR values) to a text file.

    The function overwrites the file with a header line, then appends one value per line.

    Args:
        val (list or array-like): List of numeric values to write.
        file (str): Path to the output file.

    Returns:
        None
    """
    data = open(file, 'w')
    data.write('SNR\n')
    data.close()
    data = open(file, 'a')
    for i in range(0, len(val)):

        data_to_write_los = '{}\n'.format(val[i])
        data.write(str(data_to_write_los))
    data.close()


def plot_scenario_2d(factory_length: float, factory_width: float, factory_height: float,
                     machine_list: List[Machine], ue_list: List[Ue], bs: BS,
                     distribution_class: Distribution, scenario_name: str, save_file: str = None, snr_list: list = None,
                     carrier_frequency_hz: list = None, x_y_list: list = None):
    """
        Plot a 2D top-down view of the factory scenario layout.

        Args:
            factory_length (float): Length of the factory [m].
            factory_width (float): Width of the factory [m].
            factory_height (float): Height of the factory [m] (unused, for consistency with 3D).
            machine_list (List[Machine]): List of Machine objects with positions and sizes.
            ue_list (List[Ue]): List of UEs objects with traffic types and positions.
            bs (BS): BS object with coordinates.
            distribution_class (Distribution): Object providing the number of machines.
            scenario_name (str): Scenario identifier for title and filenames.
            save_file (str, optional): Path prefix to save the generated plot.
            snr_list (list, optional): Optional list of SNR values for visualization.
            carrier_frequency_hz (list, optional): Optional frequency list (for labeling).
            x_y_list (list, optional): Optional (x, y) coordinates to overlay SNR distribution.

        Returns:
            None
    """

    plt.plot((0, 0), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (0, 0), color='red')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (factory_width, factory_width), color='red')
    legend_items = list()
    legend_labels = list()
    index_ct = 0

    for i in range(distribution_class.get_number_of_machines()):
        legend_item = plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='gray', s=10)
        if i == 1:
            legend_items.append(legend_item)
            legend_labels.append('Machine Center')
        machine_size = machine_list[i].get_machine_size()
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='gray')
        legend2 = plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                           [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                           color='gray')

        # plt.text(machine_list[i].x_center, machine_list[i].y_center, f'{i}', fontsize=10, ha='right')


        for index, ue in enumerate(ue_list):
            if ue.traffic_type == 'traffic_rt':
                legend3 = plt.scatter(ue.x, ue.y, color='red')

            elif ue.traffic_type == 'traffic_nrt':
                legend4 = plt.scatter(ue.x, ue.y, color='green')

            elif ue.traffic_type == 'traffic_cn':
                legend5 = plt.scatter(ue.x, ue.y, color='c')

            elif ue.traffic_type == 'traffic_fq':
                legend_ue = plt.scatter(ue.x, ue.y, color='black', s=10)
                index_ct += 1

                # Add the legend only one time
                if index_ct == 1:
                    legend_items.append(legend_ue)
                    legend_labels.append('UE')


            # Add text box with UE ID
            # plt.text(ue.x, ue.y, f'{ue.get_ue_id()}', fontsize=10, ha='right')
        # plt.text(machine_list[i].x_center, machine_list[i].y_center, f'{i}', fontsize=10, ha='right')
    # BS plot
    legend_bs = plt.scatter(bs.x, bs.y, s=40, color='blue', marker="^")
    legend_items.append(legend_bs)
    legend_labels.append('BS')
    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels, loc="lower left", fontsize=10, ncol=1, bbox_to_anchor=(0.33, 0.952, -1.5, 1))

    # Get current date
    current_date = datetime.now()

    # Format the date components
    year = current_date.year
    month = current_date.month
    day = current_date.day

    # Save file
    if save_file is not None:
        final_file_name_plot = save_file + f'{year}_{month:02d}_{day:02d}_' + f'plot_scenario_{scenario_name}_2d.png'
        plt.savefig(final_file_name_plot, dpi=300)
        print('Output plot saved in {}'.format(final_file_name_plot))
    plt.show()

