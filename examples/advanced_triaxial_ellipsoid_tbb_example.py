"""
NOTE: This advanced example requires GSL and TBB, and as such, it is currently not supported by Windows and macOS.
See https://github.com/geoffreygarrett/pydin/issues/1 for more information.
"""
import pydin.core.logging as pdlog
import pydin.core.tbb as tbb

# Import reused components from basic example
from basic_triaxial_ellipsoid_example import initialize_gravity, create_meshgrid, create_contour_plot, ModelParams


def calculate_potential_with_tbb(gravity, X, Y, Z, parallelism):
    """Calculates the potential using the provided gravity model with TBB."""
    timer_name = f"Gravitational potential calculation with TBB (parallelism: {parallelism})"
    pdlog.start_timer(timer_name)
    with tbb.TBBControl() as tbb_ctrl:
        tbb_ctrl.max_allowed_parallelism = parallelism
        U = gravity.calculate_potentials(X, Y, Z)
    elapsed_time = pdlog.stop_timer(timer_name)
    return U, elapsed_time


def run_advanced_example():
    pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting advanced tri-axial ellipsoid example with TBB")

    # Initialize parameters
    params = ModelParams(a=300.0, b=200.0, c=100.0, rho=2.8 * 1000.0, limit=1000.0, n=1000, z=0.0)

    # Initialize gravitational model
    gravity = initialize_gravity(params)

    # Create meshgrid
    X, Y, Z = create_meshgrid(params)

    # Set different levels of parallelism
    parallelism_levels = [
        tbb.hardware_concurrency(),
        int(0.75 * tbb.hardware_concurrency()),
        int(0.5 * tbb.hardware_concurrency()),
        int(0.25 * tbb.hardware_concurrency()),
        1
    ]

    # File to record the differences
    with open('differences.txt', 'w') as file:
        # Calculate potential with TBB at different levels of parallelism and record the differences
        for parallelism in parallelism_levels:
            U, elapsed_time = calculate_potential_with_tbb(gravity, X, Y, Z, parallelism)
            file.write(f'Parallelism: {parallelism}, Elapsed time: {elapsed_time}\n')

            # Create contour plot
            create_contour_plot(X, Y, U, params, filename=f'gravitational_potential_tbb_{parallelism}.png')

    pdlog.info("Finished advanced tri-axial ellipsoid example with TBB, goodbye!")


if __name__ == '__main__':
    run_advanced_example()
