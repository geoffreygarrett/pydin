import importlib
import timeit

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


def time_func(lib_func, num_runs, single_input, size, params):
    lib_module = lib_func.__module__
    if single_input:
        time = timeit.timeit(lambda: lib_func(1., 5., size, **params), number=num_runs) / num_runs
    else:
        size_sqrt = int(size ** 0.5)
        linspace_func = getattr(importlib.import_module(lib_module), "linspace")
        input_data = [linspace_func(1., 5., size_sqrt, **params)] * 2
        time = timeit.timeit(lambda: lib_func(*input_data), number=num_runs) / num_runs
    return time


def plot_results(benchmarks, sizes, save_as=None, show=True, **kwargs):
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


def run_benchmarks(lib_funcs_dict, single_input=True, num_runs=10, start=1, end=6, num_points=20, save_as=None,
                   show=True, use_error_bars=False, num_repeats=3, **kwargs):
    sizes = np.logspace(start, end, num_points).astype(int)

    # Create a benchmark structure with either lists or dicts based on use_error_bars
    if use_error_bars:
        benchmarks = {lib: {func: {str(params): {'mean': [], 'std': []} for params in params_list}
                            for func, params_list in funcs.items()}
                      for lib, funcs in lib_funcs_dict.items()}
    else:
        benchmarks = {lib: {func: {str(params): [] for params in params_list}
                            for func, params_list in funcs.items()}
                      for lib, funcs in lib_funcs_dict.items()}

    for size in sizes:
        for lib, funcs in lib_funcs_dict.items():
            lib_module = importlib.import_module(lib)
            for func, params_list in funcs.items():
                lib_func = getattr(lib_module, func)
                for params in params_list:
                    times = []
                    for _ in range(num_repeats if use_error_bars else 1):
                        times.append(time_func(lib_func, num_runs, single_input, size, params))
                    if use_error_bars:
                        benchmarks[lib][func][str(params)]['mean'].append(np.mean(times))
                        benchmarks[lib][func][str(params)]['std'].append(np.std(times))
                    else:
                        benchmarks[lib][func][str(params)].append(np.mean(times))

    plot_results(benchmarks, sizes, save_as, show, **kwargs)
