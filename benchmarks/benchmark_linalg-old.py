from utils import run_benchmarks

single_input_libs = {
    'numpy': {
        'linspace': [{}],
        'logspace': [{}],
        'geomspace': [{}]
    },
    'pydin.core.linalg': {
        'linspace': [{'parallel': True}, {'parallel': False}],
        'logspace': [{'parallel': True}, {'parallel': False}],
        'geomspace': [{'parallel': True}, {'parallel': False}]
    },
    'pydin.core.linalg.eigen': {
        'linspace': [{}],
    },
}

multi_input_libs = {
    'numpy': {'meshgrid': [{}]},
    'pydin.core.linalg': {'meshgrid': [{'parallel': True}, {'parallel': False}]},
    'pydin.core.linalg.eigen': {'meshgrid': [{}]}
}

custom_plot_settings = {
    'fill_between_color': 'lightblue',
    'fill_between_alpha': 0.3,
    'linestyles': ['-', ':', '--', '-.', (0, (1, 10))],
    'xlabel': 'Array Size (log scale)',
    'ylabel': 'Execution Time (s, log scale)',
    'title': 'Performance Benchmarking',
    'figure_size': (15, 10),
}

run_benchmarks(single_input_libs,
               single_input=True,
               save_as='benchmark_linalg_single_input.png',
               show=False, end=6,
               num_points=20,
               use_error_bars=True, **custom_plot_settings)

run_benchmarks(multi_input_libs,
               single_input=False,
               save_as='benchmark_linalg_multi_input.png',
               show=False,
               end=6,
               num_points=20,
               use_error_bars=True, **custom_plot_settings)
