import numpy as np

import benchmarks as bench
from pydin import attempt_import
from utils import run_benchmarks

LINSPACE_NORMAL = 'linspace_normal'
LINSPACE_PARALLEL = 'linspace_parallel'
LINSPACE = 'linspace'


@bench.fixture(name=LINSPACE_NORMAL, params={
    'start': 0,
    'stop': 1,
    'num': bench.range(50, 1000, 50),
    'endpoint': True,
    'retstep': False,
})
def linspace_fixture(request):
    return request.x


@bench.register(
    tags=[LINSPACE_NORMAL],
    name='np.linspace',
)
def np_linspace_benchmark(request):
    # this will be run before timing
    # use this for numba.jit, numba.njit, warmup, etc.
    request.startup = lambda: eval("""
    import numpy as np
    """)

    # only this will be timed
    request.execute = lambda: eval("""
    np.linspace(**request.fixture_params)
    """"")

    # this will be run after timing
    request.teardown = lambda: eval("""
    del np
    """)


@bench.register(
    tags=[LINSPACE_NORMAL],
    name='pydin.linspace',
)
def np_linspace_benchmark(request):
    # this will be run before timing
    # use this for numba.jit, numba.njit, warmup, etc.
    request.startup = lambda: eval("""
    import pydin.core.linalg as np
    """)

    # only this will be timed
    request.execute = lambda: eval("""
    np.linspace(**request.params)
    """"")

    # this will be run after timing
    request.teardown = lambda: eval("""
    del np
    """)


@bench.results(tags=[LINSPACE], expected=np.ndarray)
def linspace_results(results):
    plt = attempt_import('matplotlib.pyplot.plt')
    fig, ax = plt.subplots(len(results), 1)
    for result in results:
        ax.plot(result.x, result.y)
    return fig, ax

# OLD BENCHMARKING METHOD:

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



namespace = [
    "numpy", lambda: attempt_import("numpy"),
    "pydin.core.linalg", lambda: attempt_import("pydin.core.linalg"),
    "pydin.core.linalg.eigen", lambda: attempt_import("pydin.core.linalg.eigen"),
    "matplotlib.pyplot", lambda: attempt_import("matplotlib.pyplot"),
]

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
