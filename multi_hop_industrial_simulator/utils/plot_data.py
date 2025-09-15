from cProfile import label
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from timessim.env.machine import Machine
from timessim.network.bs import BS
from timessim.network.ris import RIS
from timessim.network.ue import Ue
from timessim.env.distribution import Distribution

font_dict = {'label_size': 22,
             'ticks_size': 22,
             'legend_size': 20}


def plot_curves(x_data, y_data, x_label, y_label, legends,
                marker=None, x_ticks=None, y_ticks=None, colors=None, show_grid=True,
                save_file=None, textbox_text=None, textbox_position: tuple = None,
                plot_type='line', hatches=None, width: float = None, ci_data=None, label_values=None,
                asymptote_y1: float = None, asymptote_y1_label: str = None, asymptote_y1_color: str = 'black', asymptote_y1_style: str = '--',
                asymptote_y2: float = None, asymptote_y2_label: str = None, asymptote_y2_color: str = 'black', asymptote_y2_style: str = '-.'):
    """
    Parameters
    ----------
    ci_data: list or numpy array
        Contains the data for the confidence interval
    x_data : list or numpy array
        Contains the data for the x axis
    y_data : list or numpy array
        Contains data for the y-axis. It can be a list of list, or a 2-dimensional array,
        if more than one curve has to be plotted.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis
    legends : list of str
        Contains the labels for the curves, with the same order of 'y_data'. If one curve
        is passed, 'legends' has to be a list with one element.
    marker : list of char, optional
        Contains the markers for the different curves.
    x_ticks : list, optional
        Contains two lists or numpy array elements, one for the tick positions, one for the
        values, for the x-axis.
    y_ticks : list, optional
        Contains two lists or numpy array elements, one for the tick positions, one for the
        values, for the y-axis.
    colors : list, optional
        List with colors for the different curves, with the same order of 'y_data'.
    show_grid : bool, optional
        Show or not the grid.
    save_file : str, optional
        Path + filename to save the plot. If None, it does not save the plot.
    textbox_text : str, optional
        Text to be inserted in the textbox
    textbox_position : str, optional
        Position of the textbox
    plot_type : str, optional
        Type of plot to generate. Either 'line' for line plots or 'bar' for bar charts.
    hatches : list, optional
        List of hatch patterns for each set of bars.
    width : float, optional
        Width of the bars.

    Returns
    -------
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
                 machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                 distribution_class: Distribution, scenario_name: str, save_file: str = None):
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


def plot_factory_paper(factory_length: float, factory_width: float, factory_height: float,
                       machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                       distribution_class: Distribution, scenario_name: str, save_file: str = None):
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
                                 color='dimgrey')

        # Add the legend only one time
        if i == 1:
            legend_items.append(legend_item)
            legend_labels.append('Machine Center')
        machine_size = machine_list[i].get_machine_size()
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center - machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center - machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],
                 [machine_list[i].z_center + machine_size / 2, machine_list[i].z_center + machine_size / 2],
                 color='dimgrey')

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
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='green', s=10)
            index_ct += 1

            # Add the legend only one time
            if index_ct == 1:
                legend_items.append(legend_item)
                legend_labels.append('UE')

    # BS plot
    legend_items.append(ax.scatter(bs.x, bs.y, bs.z, s=40, color='k', marker="^"))
    legend_labels.append('BS')

    # Insert title
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)

    # Inser labels
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    ax.set_zlabel('Factory height [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels)

    # Save file
    if save_file is not None:
        plt.savefig(final_file_name_plot, dpi=300)
        print('Output plot saved in {}'.format(final_file_name_plot))

    # Show the plot
    plt.show()


def plot_snr(factory_length: float, factory_width: float, factory_height: float,
             machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
             distribution_class: Distribution, scenario_name: str, apply_fading: bool, fading: float, shad: float,
             pt: float, save_file: str = None, snr_list: list = None,
             carrier_frequency_hz: list = None, ue_list_x: list = None, ue_list_y: list = None):
    # fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot((0, 0), (factory_width, 0), color='grey')
    plt.plot((0, factory_length), (0, 0), color='grey')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='grey')
    plt.plot((0, factory_length), (factory_width, factory_width), color='grey')

    legend_very_low = 0
    legend_low = 0
    legend_medium = 0
    legend_high = 0
    legend_very_high = 0

    legend_items = []
    legend_labels = []
    index_ct = 0

    for i in range(0, len(machine_list)):
        # plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='dimgrey')
        machine_size = machine_list[i].get_machine_size()
        legend_m = plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                            [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                            color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)

    for k in range(len(snr_list)):

        if snr_list[k] >= 7:  # 10:  # < minimum + 3 * lev:

            legend_item0 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightsteelblue') #skyblue
            legend_very_high += 1
            if legend_very_high == 1:
                legend_items.append(legend_item0)
                legend_labels.append('$SNR \geq 7$ dB')

        elif 7 > snr_list[k] >= 5:  # 10 and snr_list[k] >= 5:  # < minimum + 2 * lev:
            legend_item1 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightskyblue') # dodgerblue
            legend_high += 1
            if legend_high == 1:
                legend_items.append(legend_item1)
                legend_labels.append('$5$ dB $\leq SNR < 7$ dB')

        elif 5 > snr_list[k] >= 2:  # <minimum+ lev:
            legend_item2 = plt.scatter(ue_list[k].x, ue_list[k].y, c='dodgerblue')  # steelblue')
            legend_medium += 1
            if legend_medium == 1:
                legend_items.append(legend_item2)
                legend_labels.append('$2$ dB $\leq SNR < 5$ dB')


        elif snr_list[k] < 2:  # and snr_list[k] >= -10:
            legend_item3 = plt.scatter(ue_list[k].x, ue_list[k].y, c='dodgerblue')
            legend_low += 1
            if legend_low == 1:
                legend_items.append(legend_item3)
                legend_labels.append('$SNR < 2$ dB')

        # if snr_list[k] >= 7:  # 10:  # < minimum + 3 * lev:
        #
        #     legend_item0 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightsteelblue') #skyblue
        #     legend_very_high += 1
        #     if legend_very_high == 1:
        #         legend_items.append(legend_item0)
        #         legend_labels.append('$SNR \geq 7$ dB')
        #
        # elif 7 > snr_list[k] >= 3.5:  # 10 and snr_list[k] >= 5:  # < minimum + 2 * lev:
        #     legend_item1 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightskyblue') # dodgerblue
        #     legend_high += 1
        #     if legend_high == 1:
        #         legend_items.append(legend_item1)
        #         legend_labels.append('$3.5 \leq SNR < 7$ dB')
        #
        # elif 3.5 > snr_list[k] >= 0:  # <minimum+ lev:
        #     legend_item2 = plt.scatter(ue_list[k].x, ue_list[k].y, c='dodgerblue')  # steelblue')
        #     legend_medium += 1
        #     if legend_medium == 1:
        #         legend_items.append(legend_item2)
        #         legend_labels.append('$0  \leq SNR < 3.5$ dB')
        #
        #
        # elif snr_list[k] < 0:  # and snr_list[k] >= -10:
        #     legend_item3 = plt.scatter(ue_list[k].x, ue_list[k].y, c='gray')
        #     legend_low += 1
        #     if legend_low == 1:
        #         legend_items.append(legend_item3)
        #         legend_labels.append('$SNR < 0$ dB')


        # elif snr_list[k] < -10:
        #   legend_item = plt.scatter(x_y_list[k][0], x_y_list[k][1], c='m')
        #  legend_very_low += 1
        # if legend_very_low == 1:
        #    legend_items.append(legend_item)
        #   legend_labels.append('SNR < -10 dB')

    for index, ue in enumerate(ue_list):

        if ue.x in ue_list_x and ue.y in ue_list_y:
            legend_ue = plt.scatter(ue.x, ue.y, color='black', s=5)

            index_ct += 1

            # Add the legend only one time
            if index_ct == 1:
                legend_items.append(legend_ue)
                legend_labels.append('UE')

    # legend0 = plt.scatter(bs.x, bs.y, s=40, color='k', marker="^")
    #    legend_items = [legend0, legend_m[0], legend_item0, legend_item1, legend_item2, legend_item3]
    # legend_labels = ['BS','$SNR \geq 10$ dB', '$5 \leq SNR < 10$ dB', '$0  \leq SNR < 5$ dB', '$-10 \leq SNR < 0$ dB']

    # legend_labels = ['BS', 'Machine side','$SNR \geq 7$ dB', '$3.5 \leq SNR < 7$ dB', '$0  \leq SNR < 3.5$ dB', '$ SNR < 0$ dB']

    # UE plot
    # if apply_fading:
    #   title = 'Factory layout ' + str(scenario_name) + ' ' + str(carrier_frequency_hz) + 'GHz, Pt=' + str(
    #   pt) + 'dBm, shad std: ' + str(shad) + ', fading std. ' + str(fading)
    # else:
    #    title = 'Factory layout ' + str(scenario_name) + ' ' + str(carrier_frequency_hz) + 'GHz, Pt=' + str(pt) + 'dBm'

    legend_bs = plt.scatter(bs.x, bs.y, s=40, color='blue', marker="^")
    legend_items.append(legend_bs)
    legend_labels.append('BS')
    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels, loc="lower left", fontsize=10, ncol=2, bbox_to_anchor=(0.11, 0.96, -1, 1))

    #    plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    # plt.legend(legend_items, legend_labels) #,  loc='upper right')

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))
    plt.show()

def plot_ps(factory_length: float, factory_width: float, factory_height: float,
             machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
             distribution_class: Distribution, scenario_name: str, apply_fading: bool, fading: float, shad: float,
             pt: float, save_file: str = None, ps_list: list = None,
             carrier_frequency_hz: list = None, ue_list_x: list = None, ue_list_y: list = None):
    # fig, ax = plt.subplots(figsize=(12, 10))
    plt.plot((0, 0), (factory_width, 0), color='grey')
    plt.plot((0, factory_length), (0, 0), color='grey')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='grey')
    plt.plot((0, factory_length), (factory_width, factory_width), color='grey')

    legend_very_low = 0
    legend_low = 0
    legend_medium = 0
    legend_high = 0
    legend_very_high = 0

    legend_items = []
    legend_labels = []
    index_ct = 0

    for i in range(0, len(machine_list)):
        # plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='dimgrey')
        machine_size = machine_list[i].get_machine_size()
        legend_m = plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                            [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],
                            color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center - machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center - machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center - machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center + machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)
        plt.plot([machine_list[i].x_center + machine_size / 2, machine_list[i].x_center + machine_size / 2],
                 [machine_list[i].y_center - machine_size / 2, machine_list[i].y_center + machine_size / 2],

                 color='dimgrey', linewidth=2.50)

    for k in range(len(ps_list)):

        if ps_list[k] >= 0.9:  # 10:  # < minimum + 3 * lev:

            legend_item0 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightsteelblue') #skyblue
            legend_very_high += 1
            if legend_very_high == 1:
                legend_items.append(legend_item0)
                legend_labels.append(r'$p_{s,link}^u \geq 0.9$')

        elif 0.9 > ps_list[k] >= 0.4:  # 10 and snr_list[k] >= 5:  # < minimum + 2 * lev:
            legend_item1 = plt.scatter(ue_list[k].x, ue_list[k].y, c='lightskyblue') # dodgerblue
            legend_high += 1
            if legend_high == 1:
                legend_items.append(legend_item1)
                legend_labels.append(r'$0.4 \leq p_{s,link}^u < 0.9$')


        elif ps_list[k] < 0.4:  # and ps_list[k] >= -10:
            legend_item3 = plt.scatter(ue_list[k].x, ue_list[k].y, c='dodgerblue')
            legend_low += 1
            if legend_low == 1:
                legend_items.append(legend_item3)
                legend_labels.append(r'$p_{s,link}^u < 0.4$')


    for index, ue in enumerate(ue_list):

        if ue.x in ue_list_x and ue.y in ue_list_y:
            legend_ue = plt.scatter(ue.x, ue.y, color='black', s=5)

            index_ct += 1

            # Add the legend only one time
            if index_ct == 1:
                legend_items.append(legend_ue)
                legend_labels.append('UE')

    # legend0 = plt.scatter(bs.x, bs.y, s=40, color='k', marker="^")
    #    legend_items = [legend0, legend_m[0], legend_item0, legend_item1, legend_item2, legend_item3]
    # legend_labels = ['BS','$SNR \geq 10$ dB', '$5 \leq SNR < 10$ dB', '$0  \leq SNR < 5$ dB', '$-10 \leq SNR < 0$ dB']

    # legend_labels = ['BS', 'Machine side','$SNR \geq 7$ dB', '$3.5 \leq SNR < 7$ dB', '$0  \leq SNR < 3.5$ dB', '$ SNR < 0$ dB']

    # UE plot
    # if apply_fading:
    #   title = 'Factory layout ' + str(scenario_name) + ' ' + str(carrier_frequency_hz) + 'GHz, Pt=' + str(
    #   pt) + 'dBm, shad std: ' + str(shad) + ', fading std. ' + str(fading)
    # else:
    #    title = 'Factory layout ' + str(scenario_name) + ' ' + str(carrier_frequency_hz) + 'GHz, Pt=' + str(pt) + 'dBm'

    legend_bs = plt.scatter(bs.x, bs.y, s=40, color='blue', marker="^")
    legend_items.append(legend_bs)
    legend_labels.append('BS')
    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels, loc="lower left", fontsize=10, ncol=2, bbox_to_anchor=(0.25, 0.93, -5, 1))

    #    plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    # plt.legend(legend_items, legend_labels) #,  loc='upper right')

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))
    plt.show()


def write_data(val, file):
    data = open(file, 'w')
    data.write('SNR\n')
    data.close()
    data = open(file, 'a')
    for i in range(0, len(val)):  # for i in range(0, len(d)): # d_values

        data_to_write_los = '{}\n'.format(val[i])
        data.write(str(data_to_write_los))
    data.close()


def plot_scenario_2d(factory_length: float, factory_width: float, factory_height: float,
                     machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                     distribution_class: Distribution, scenario_name: str, save_file: str = None, snr_list: list = None,
                     carrier_frequency_hz: list = None, x_y_list: list = None):
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


def plot_scenario_2d2(factory_length: float, factory_width: float, factory_height: float,
                      machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                      distribution_class: Distribution, scenario_name: str, save_file: str = None,
                      snr_list: list = None,
                      carrier_frequency_hz: list = None, x_y_list: list = None):
    plt.plot((0, 0), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (0, 0), color='red')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (factory_width, factory_width), color='red')

    for i in range(distribution_class.get_number_of_machines()):
        legend0 = plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='gray')
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

        legend1 = plt.scatter(bs.x, bs.y, s=40, color='black', marker="^")
        # plt.text(machine_list[i].x_center, machine_list[i].y_center, f'{i}', fontsize=10, ha='right')

        for index, ue in enumerate(ue_list):
            if ue.traffic_type == 'traffic_rt':
                legend3 = plt.scatter(ue.x, ue.y, color='red')

            elif ue.traffic_type == 'traffic_nrt':
                legend4 = plt.scatter(ue.x, ue.y, color='green')

            elif ue.traffic_type == 'traffic_cn':
                legend5 = plt.scatter(ue.x, ue.y, color='c')

            elif ue.traffic_type == 'traffic_fq':
                legend5 = plt.scatter(ue.x, ue.y, color='green')

            # Add text box with UE ID
            # plt.text(ue.x, ue.y, f'{ue.get_ue_id()}', fontsize=10, ha='right')
        # plt.text(machine_list[i].x_center, machine_list[i].y_center, f'{i}', fontsize=10, ha='right')

    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')

    plt.legend((legend0, legend5, legend1),
               ('Machine Center', 'UE', 'BS'))

    #    plt.legend((legend0, legend3, legend5, legend2, legend4, legend1),
    #            ('Machine Center', 'UE RT', 'UE CN', 'machines', 'UE NRT', 'BS',))

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))
    plt.show()


def plot_factory_los_nlos(factory_length: float, factory_width: float, factory_height: float,
                          machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                          distribution_class: Distribution, scenario_name: str, save_file: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

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
                                 color='gray')

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
        if ue.is_in_los == True:
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='green')
            plt.plot([ue.x, bs.x], [ue.y, bs.y], [ue.z, bs.z], color='green')

        elif ue.is_in_los == False:
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='red')
            plt.plot([ue.x, bs.x], [ue.y, bs.y], [ue.z, bs.z], color='red')

    # BS plot
    legend_items.append(ax.scatter(bs.x, bs.y, bs.z, s=40, color='black', marker="^"))
    legend_labels.append('BS')

    # Insert title
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)

    # Inser labels
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    ax.set_zlabel('Factory height [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels, loc='upper right', bbox_to_anchor=(0.5, 0.5))

    # loc="lower left", fontsize=24,
    #         mode="expand", ncol=2, bbox_to_anchor=(0, 0.97, 1, 0.1))

    # Save file
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))

    # Show the plot
    plt.show()


def plot_factory_los_nlos_ue_ue(factory_length: float, factory_width: float, factory_height: float,
                                machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                                distribution_class: Distribution, scenario_name: str, k: int, save_file: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

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
                                 color='gray')

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
        if ue == ue_list[k]:
            legend_item = ax.scatter(ue.x, ue.y, ue.z, color='black')
        elif ue != ue_list[k]:
            if ue.is_in_los_ues[k] == True:
                legend_item = ax.scatter(ue.x, ue.y, ue.z, color='green')

                plt.plot([ue.x, ue_list[k].x], [ue.y, ue_list[k].y], [ue.z, ue_list[k].z], color='green')

            elif ue.is_in_los_ues[k] == False:
                legend_item = ax.scatter(ue.x, ue.y, ue.z, color='red')
                plt.plot([ue.x, ue_list[k].x], [ue.y, ue_list[k].y], [ue.z, ue_list[k].z], color='red')


    # BS plot
    legend_items.append(ax.scatter(bs.x, bs.y, bs.z, s=40, color='black', marker="^"))
    legend_labels.append('BS')

    # Insert title
    title = 'Factory layout ' + str(scenario_name)
    # plt.title(title)

    # Inser labels
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    ax.set_zlabel('Factory height [m]')

    # Insert legend
    plt.legend(legend_items, legend_labels)

    # Save file
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))

    # Show the plot
    plt.show()


def plot_scenario_2d_los_nlos(factory_length: float, factory_width: float, factory_height: float,
                              machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                              distribution_class: Distribution, scenario_name: str, save_file: str = None,
                              snr_list: list = None,
                              carrier_frequency_hz: list = None, x_y_list: list = None):
    plt.plot((0, 0), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (0, 0), color='red')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (factory_width, factory_width), color='red')

    for i in range(distribution_class.get_number_of_machines()):
        legend0 = plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='gray')
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

        legend1 = plt.scatter(bs.x, bs.y, s=40, color='blue', marker="^")

        for index, ue in enumerate(ue_list):

            if ue.is_in_los == True:
                plt.plot([bs.x, ue.x], [bs.y, ue.y], color='green')
                legend_item = plt.scatter(ue.x, ue.y, color='green')



            elif ue.is_in_los == False:

                plt.plot([bs.x, ue.x], [bs.y, ue.y], color='red')
                legend_item = plt.scatter(ue.x, ue.y, color='red')

    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    # plt.legend((legend0, legend3, legend5, legend2, legend4, legend1),
    #           ('Machine Center', 'UE RT', 'UE CN', 'machines', 'UE NRT', 'BS',))

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))
    plt.show()


def plot_scenario_2d_los_nlos_ue_ue(factory_length: float, factory_width: float, factory_height: float,
                                    machine_list: List[Machine], ue_list: List[Ue], ris_list: List[RIS], bs: BS,
                                    distribution_class: Distribution, scenario_name: str, k: int, save_file: str = None,
                                    snr_list: list = None,
                                    carrier_frequency_hz: list = None, x_y_list: list = None):
    plt.plot((0, 0), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (0, 0), color='red')
    plt.plot((factory_length, factory_length), (factory_width, 0), color='red')
    plt.plot((0, factory_length), (factory_width, factory_width), color='red')

    for i in range(distribution_class.get_number_of_machines()):
        legend0 = plt.scatter(machine_list[i].x_center, machine_list[i].y_center, color='gray')
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

        legend1 = plt.scatter(bs.x, bs.y, s=40, color='blue', marker="^")

        for index, ue in enumerate(ue_list):

            if ue == ue_list[k]:
                legend_item = plt.scatter(ue.x, ue.y, color='black')
            elif ue != ue_list[k]:
                if ue.is_in_los_ues[k] == True:
                    legend_item = plt.scatter(ue.x, ue.y, color='green')
                    plt.plot([ue_list[k].x, ue.x], [ue_list[k].y, ue.y], color='green')

                elif ue.is_in_los_ues[k] == False:
                    legend_item = plt.scatter(ue.x, ue.y, color='red')
                    plt.plot([ue_list[k].x, ue.x], [ue_list[k].y, ue.y], color='red')

            # if ue.is_in_los == True:
            #  legend_item = plt.scatter(ue.x, ue.y, ue.z, color='green')

        # elif ue.is_in_los == False:
        #    legend_item = plt.scatter(ue.x, ue.y, ue.z, color='red')

    # UE plot
    title = 'Factory layout ' + str(scenario_name)
    plt.title(title)
    plt.xlabel('Factory length [m]')
    plt.ylabel('Factory width [m]')
    # plt.legend((legend0, legend3, legend5, legend2, legend4, legend1),
    #           ('Machine Center', 'UE RT', 'UE CN', 'machines', 'UE NRT', 'BS',))

    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        print('Output plot saved in {}'.format(save_file))
    plt.show()
