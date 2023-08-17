import matplotlib.pyplot as plt
import numpy as np

# Define a global dictionary to store common plot settings.
global_plot_settings = {
    'fill_between_color': 'lightgray',
    'fill_between_alpha': 0.5,
    'linestyles': ['-', '--', ':', '-.', (0, (1, 10))],
    'xlabel': 'Size',
    'ylabel': 'Time (s)',
    'title': 'Performance Comparison',
    'figure_size': (12, 8),
}


def plot_benchmark_results(benchmarks, sizes, save_as=None, show=True, **kwargs):
    plt.figure(figsize=kwargs.get('figure_size', global_plot_settings['figure_size']))
    linestyles = kwargs.get('linestyles', global_plot_settings['linestyles'])

    for idx, (lib, func_times) in enumerate(benchmarks.items()):
        linestyle = linestyles[idx % len(linestyles)]
        for func, params_times in func_times.items():
            for params, times in params_times.items():
                if isinstance(times, dict):  # we have error bars
                    lower_bound = np.array(times['mean']) - np.array(times['std'])
                    upper_bound = np.array(times['mean']) + np.array(times['std'])
                    fill_color = kwargs.get('fill_between_color', global_plot_settings['fill_between_color'])
                    fill_alpha = kwargs.get('fill_between_alpha', global_plot_settings['fill_between_alpha'])
                    plt.fill_between(sizes, lower_bound, upper_bound, color=fill_color, alpha=fill_alpha)
                    plt.plot(sizes, times['mean'], label=f'{lib} {func} {str(params)}', linestyle=linestyle)

                else:  # plot normally
                    plt.plot(sizes, times, label=f'{lib} {func} {str(params)}', linestyle=linestyle)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(kwargs.get('xlabel', global_plot_settings['xlabel']))
    plt.ylabel(kwargs.get('ylabel', global_plot_settings['ylabel']))
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.title(kwargs.get('title', global_plot_settings['title']))

    if save_as:
        plt.savefig(save_as, dpi=300)

    if show:
        plt.show()
