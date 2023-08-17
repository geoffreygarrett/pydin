# import benchmarks as bench
import textwrap

import matplotlib.pyplot as plt
import numpy as np

import benchmarks as bench
from pydin import logging

logger = logging.stdout_color_mt(__name__)


# LINSPACE ############################################################################################################
@bench.fixture(name="linspace", params={
    'start': bench.SingleParameter(0),
    'stop': bench.SingleParameter(1),
    'num': bench.ArrayParameter(np.logspace(3, 8, 30).astype(np.int64))
})
def linspace_fixture(params):
    return params


@bench.benchmark(tags=['linspace'], name='numpy.linspace')
def benchmark_numpy_linspace(request):
    request.startup = textwrap.dedent("""
    import numpy as np
    """)

    request.execute = textwrap.dedent(f"""
    np.linspace(**params)
    """)


@bench.benchmark(tags=['linspace'], name='pydin.linspace')
def benchmark_pydin_linspace(request):
    request.startup = textwrap.dedent("""
    import pydin
    """)
    request.execute = textwrap.dedent(f"""
    pydin.core.linalg.linspace(**params)
    """)


@bench.benchmark(tags=['linspace'], name='numba.jit[warm]')
def benchmark_numba_jit_linspace(request):
    request.startup = textwrap.dedent("""
    import numba as nb
    import numpy as np 
    
    @nb.jit(nopython=True)
    def linspace(start, stop, num):
        return np.linspace(start, stop, num)
        
    _ = linspace(0, 1, 100) # WARMUP
    """)
    request.execute = textwrap.dedent(f"""
    linspace(**params)
    """)


# MESHGRID ############################################################################################################
@bench.fixture(name='meshgrid', params={
    'start': bench.SingleParameter(0),
    'stop': bench.SingleParameter(1),
    'num': bench.ArrayParameter(np.logspace(3, 5, 30).astype(np.int64)),
})
def meshgrid_fixture(params):
    return params


@bench.benchmark(tags=['meshgrid'], name='pydin.meshgrid')
def benchmark_pydin_meshgrid(request):
    request.startup = textwrap.dedent("""
    import pydin
    x = pydin.core.linalg.linspace(**params)
    y = pydin.core.linalg.linspace(**params)
    """)
    request.execute = textwrap.dedent(f"""
    pydin.core.linalg.meshgrid(x, y)
    """)


@bench.benchmark(tags=['meshgrid'], name='numpy.meshgrid')
def benchmark_numpy_meshgrid(request):
    request.startup = textwrap.dedent("""
    import numpy as np
    x = np.linspace(**params)
    y = np.linspace(**params)
    """)
    request.execute = textwrap.dedent(f"""
    np.meshgrid(x, y)
    """)


import pandas as pd

# @bench.analysis(name='linspace', tags=['linspace'])
# def analysis_linspace(results):
#     # `results` is now a list of BenchmarkResult
#     avg_times = [r.avg_time for r in results]
#     std_devs = [np.sqrt(r.variance) for r in results]
#     names = [r.name for r in results]
#     num_params = [r.params['num'] for r in results]  # Assuming 'num' is stored in 'params'
#
#     # Create a pandas DataFrame from the results
#     df = pd.DataFrame({
#         'Benchmark': names,
#         'Average Time (s)': avg_times,
#         'Std Deviation': std_devs,
#         'Num Parameter': num_params
#     })
#
#     # Create a figure and a set of subplots
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Plot average times against 'num' parameter for each benchmark with error bars
#     for name in df['Benchmark'].unique():
#         df_name = df[df['Benchmark'] == name]
#         ax.errorbar(df_name['Num Parameter'], df_name['Average Time (s)'], yerr=df_name['Std Deviation'], fmt='o', label=name)
#
#     # Set labels, title, and legend
#     ax.set_xlabel('Num Parameter')
#     ax.set_ylabel('Average Time (s)')
#     ax.set_title('Average Time against Num Parameter for Each Benchmark')
#     ax.legend()
#
#     # Automatically adjust the subplot params so the subplot fits into the figure area
#     fig.tight_layout()
#     fig.savefig('linspace.png')
import itertools


@bench.analysis(name='linalg', tags=['linspace', 'meshgrid'])
def analysis_linspace(all_results):
    logger.info(f"Starting analysis for linalg")

    # Log the content of all_results
    logger.debug(f"Received all_results: {all_results}")

    for tag, results in all_results.items():
        logger.info(f"Processing results for tag: {tag}")

        # # This will log the number of results for the current tag
        logger.debug(f"Number of results for tag {tag}: {len(results)}")

        # If you also want to log each result:
        for i, result in enumerate(results):
            logger.debug(f"Result {i} for tag {tag}: {result}")

        # `results` is now a list of BenchmarkResult
        avg_times = [r.avg_time for r in results]
        std_devs = [np.sqrt(r.variance) for r in results]
        names = [r.name for r in results]
        num_params = [r.params['num'] for r in results]  # Assuming 'num' is stored in 'params'

        # Create a pandas DataFrame from the results
        df = pd.DataFrame({
            'Benchmark': names,
            'Average Time (s)': avg_times,
            'Std Deviation': std_devs,
            'Num Parameter': num_params
        })

        # Define plot settings
        plot_settings = {
            'fill_between_alpha': 0.2,
            'linestyles': ['-', '--', ':', '-.', (0, (1, 10))],
            'markers': ['o', 's', 'v', '^', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_', '.'],
            'colors': plt.get_cmap('tab10').colors,  # use a colormap with 10 distinct colors
            'xlabel': 'Num Parameter',
            'ylabel': 'Average Time (s)',
            'title': 'Average Time against Num Parameter for Each Benchmark',
            'figure_size': (12, 8),
        }

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=plot_settings['figure_size'])

        # Create a marker style cycle (will cycle if number of benchmarks is more than marker styles)
        marker_cycle = itertools.cycle(plot_settings['markers'])

        # Plot average times against 'num' parameter for each benchmark with error bars
        for idx, name in enumerate(df['Benchmark'].unique()):
            linestyle = plot_settings['linestyles'][idx % len(plot_settings['linestyles'])]
            marker = next(marker_cycle)
            color = plot_settings['colors'][idx % len(plot_settings['colors'])]
            df_name = df[df['Benchmark'] == name]
            lower_bound = df_name['Average Time (s)'] - df_name['Std Deviation']
            upper_bound = df_name['Average Time (s)'] + df_name['Std Deviation']
            ax.fill_between(df_name['Num Parameter'], lower_bound, upper_bound, color=color,
                            alpha=plot_settings['fill_between_alpha'])
            ax.errorbar(df_name['Num Parameter'], df_name['Average Time (s)'], yerr=df_name['Std Deviation'],
                        fmt=marker,
                        linestyle=linestyle, color=color, label=name, capsize=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(plot_settings['xlabel'])
        ax.set_ylabel(plot_settings['ylabel'])
        ax.set_title(plot_settings['title'])
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place the legend outside of the plot to improve clarity

        # Automatically adjust the subplot params so the subplot fits into the figure area
        fig.tight_layout()
        fig.savefig(f'{tag}.png', bbox_inches='tight')  # Ensure the outside legend is included in the saved figure


# @bench.analysis(name='linspace', tags=['linspace'])
# def analysis_linspace(results):
#     # `results` is now a list of BenchmarkResult
#     # Access time, variance, and extra results like so:
#     times = [r.total_time for r in results]
#     variances = [r.variance for r in results]
#     names = [r.name for r in results]
#
#     # Create a figure and a set of subplots
#     fig, ax1 = plt.subplots()
#
#     # Define the width of the bars
#     bar_width = 0.35
#
#     # Create an index for each group of bars
#     x = np.arange(len(names))
#
#     # Plot the times
#     rects1 = ax1.bar(x - bar_width / 2, times, bar_width, label='Total Time', color='b')
#
#     # Create a second y-axis for the variance
#     ax2 = ax1.twinx()
#
#     # Plot the variances
#     rects2 = ax2.bar(x + bar_width / 2, variances, bar_width, label='Variance', color='r')
#
#     # Add labels, title, and legend
#     ax1.set_xlabel('Benchmark')
#     ax1.set_ylabel('Total Time (s)', color='b')
#     ax2.set_ylabel('Variance (s²)', color='r')
#     ax1.set_title('Linspace Benchmark Results')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(names)
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
#
#     # Automatically adjust the subplot params so the subplot fits into the figure area
#     fig.tight_layout()
#
#     # Display the plot
#     plt.savefig('linspace.png')


if __name__ == "__main__":
    logging.set_level(logging.INFO)
    runner = bench.BenchmarkRunner()
    # runner.clear_cache()
    runner.run_benchmarks(tags=["meshgrid", "linspace"])
    runner.run_analysis("linalg")
