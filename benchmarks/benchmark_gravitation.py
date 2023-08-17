import textwrap

import numpy as np
import pydin.core.logging as logging

logger = logging.stdout_color_mt(__name__)

import benchmarks as bench


@bench.fixture(name='gravity', params={
    'limit': bench.SingleParameter(1000),
    'subdivisions': bench.SingleParameter(0),
    'num': bench.ArrayParameter(np.logspace(1, 2.5, 15, dtype=int)),
    'a': bench.SingleParameter(300.0),
    'b': bench.SingleParameter(200.0),
    'c': bench.SingleParameter(100.0),
    'rho': bench.SingleParameter(2.8 * 1000.0),
    'G': bench.SingleParameter(6.67408 * 1e-11),
    'mu': bench.SingleParameter(4.0 / 3.0 * np.pi * 2.8 * 1000.0 * 6.67408 * 1e-11 * 300.0 * 200.0 * 100.0)
})
def gravity_fixture(params):
    return params


@bench.benchmark(tags=['gravity', 'tri_axial_ellipsoid'], name='TriAxialEllipsoid.acceleration')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    from pydin.core.gravitation import TriAxialEllipsoid
    model = TriAxialEllipsoid(a=params['a'], b=params['b'], c=params['c'], mu=params['mu'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    xv, yv = np.meshgrid(x, y)
    positions = np.array([xv.flatten(), yv.flatten(), np.zeros(xv.size)]).T
    """)
    request.execute = textwrap.dedent(f"""
    for position in positions:
        model.acceleration(position)
    """)


@bench.benchmark(tags=['gravity', 'tri_axial_ellipsoid'], name='TriAxialEllipsoid.acceleration_series')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    from pydin.core.gravitation import TriAxialEllipsoid
    model = TriAxialEllipsoid(a=params['a'], b=params['b'], c=params['c'], mu=params['mu'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    xv, yv = np.meshgrid(x, y)
    positions = np.array([xv.flatten(), yv.flatten(), np.zeros(xv.size)]).T
    """)
    request.execute = textwrap.dedent(f"""
    model.acceleration_series(positions)
    """)


@bench.benchmark(tags=['gravity', 'tri_axial_ellipsoid'], name='TriAxialEllipsoid.acceleration_grid')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    from pydin.core.gravitation import TriAxialEllipsoid
    model = TriAxialEllipsoid(a=params['a'], b=params['b'], c=params['c'], mu=params['mu'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    positions = np.meshgrid(x, y, [0.0], indexing='ij')
    """)
    request.execute = textwrap.dedent(f"""
    model.acceleration_grid(positions)
    """)


@bench.benchmark(tags=['gravity', 'icosphere_ellipsoid'], name='Polyhedral.acceleration[icosphere1]')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    import trimesh
    from pydin.core.gravitation import Polyhedral
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    radii = np.array([params['a'], params['b'], params['c']])
    ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))
    model = Polyhedral(nodes=ellipsoid_mesh.vertices, faces=ellipsoid_mesh.faces, density=params['rho'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    xv, yv = np.meshgrid(x, y)
    positions = np.array([xv.flatten(), yv.flatten(), np.zeros(xv.size)]).T
    """)
    request.execute = textwrap.dedent(f"""
    for position in positions:
        model.acceleration(position)
        
    """)


@bench.benchmark(tags=['gravity', 'icosphere_ellipsoid'], name='Polyhedral.acceleration_series[icosphere1]')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    import trimesh
    from pydin.core.gravitation import Polyhedral
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    radii = np.array([params['a'], params['b'], params['c']])
    ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))
    model = Polyhedral(nodes=ellipsoid_mesh.vertices, faces=ellipsoid_mesh.faces, density=params['rho'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    xv, yv = np.meshgrid(x, y)
    positions = np.array([xv.flatten(), yv.flatten(), np.zeros(xv.size)]).T
    """)
    request.execute = textwrap.dedent(f"""
    model.acceleration_series(positions)
    """)


@bench.benchmark(tags=['gravity', 'icosphere_ellipsoid'], name='Polyhedral.acceleration_grid[icosphere1]')
def benchmark_gravity(request):
    request.startup = textwrap.dedent(f"""
    import numpy as np
    import trimesh
    from pydin.core.gravitation import Polyhedral
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    radii = np.array([params['a'], params['b'], params['c']])
    ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))
    model = Polyhedral(nodes=ellipsoid_mesh.vertices, faces=ellipsoid_mesh.faces, density=params['rho'])
    limit = params['limit']
    num = params['num']
    x = np.linspace(-limit, limit, num)
    y = np.linspace(-limit, limit, num)
    positions = np.meshgrid(x, y, [0.0], indexing='ij')
    """)
    request.execute = textwrap.dedent(f"""
    model.acceleration_grid(positions)
    """)


# # Define the radii of the ellipsoid
# radii = np.array([3.0, 2.0, 1.0])
#
# # Create a unit sphere
# sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
#
# # Scale it to form the ellipsoid
# ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))

# nodes = ellipsoid_mesh.vertices
# faces = ellipsoid_mesh.faces


import pandas as pd
import itertools
import matplotlib.pyplot as plt


@bench.analysis(name='gravity', tags=['gravity'])
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
        print(results)
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


if __name__ == "__main__":
    logging.set_level(logging.INFO)
    runner = bench.BenchmarkRunner(min_trials=10, min_time=1.0)
    runner.clear_cache()
    runner.run_benchmarks(tags=["gravity"])
    runner.run_analysis("gravity")
