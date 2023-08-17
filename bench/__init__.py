import abc
import functools
import time

import numpy as np

import utils as bu
from pydin import get_logger

logger = get_logger(__name__)

__all__ = [
    'ParameterSet',
    'range',
    'values',
    'boolean',
    'fixture',
    'return_fixture',
    'register',
]


class ParameterSet:
    def __init__(self, **kwargs):
        self.params = kwargs


def range(start, stop=None, step=None, num=None):
    if stop is None and step is None and num is None:
        # If only one argument is given, interpret it as a built-in range
        return ParameterSet(values=list(range(start)))
    elif stop is not None and num is not None:
        # If stop and num are given, use them with start to create a linspace
        return ParameterSet(values=np.linspace(start, stop, num).tolist())
    elif isinstance(start, np.ndarray):
        # If a numpy array is given, use it directly
        return ParameterSet(values=start.tolist())
    else:
        # Otherwise, use the arguments to create a list of values
        return ParameterSet(values=list(np.arange(start, stop, step)))


def values(*args):
    return ParameterSet(values=list(args))


def boolean():
    return ParameterSet(values=[True, False])


def _pydin_detail(func):
    """
    Decorator to add detailed logging, timing, and error handling to a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global logger
        # logger = logging.get_logger(func.__name__)
        # if logger is None:
        #     logger = get_logger(func.__name__)
        logger.info(f"Starting {func.__name__} with arguments {args} and keyword arguments {kwargs}.")

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"An error occurred in {func.__name__} : {str(e)}")
            raise
        else:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info(f"Finished {func.__name__} in {elapsed_time:.4f} seconds with result: {result}.")
            return result

    return wrapper


# Store registered benchmarks as a dictionary
_BENCHMARKS = {}

# Store registered fixtures
_FIXTURES = {}

# Store return fixtures
_RETURN_FIXTURES = {}


@_pydin_detail
def fixture(func):
    """
    Decorator to register fixture functions.
    """
    _FIXTURES[func.__name__] = func
    return func


@_pydin_detail
def return_fixture(func):
    """
    Decorator to register return fixtures.
    """
    _RETURN_FIXTURES[func.__name__] = func
    return func


@_pydin_detail
def register(func=None, *,
             output_format="xml",
             output_location=".",
             fixture_name=None,
             return_fixture_name=None):
    """
    Decorator to register benchmark functions or classes with optional configurations.
    """

    def decorator_benchmark(f):
        config = {'output_format': output_format, 'output_location': output_location}

        fixture_params = None
        if fixture_name is not None:
            fixture_func = _FIXTURES.get(fixture_name)
            if fixture_func is not None:
                fixture_params = fixture_func()

        return_fixture_func = None
        if return_fixture_name is not None:
            return_fixture_func = _RETURN_FIXTURES.get(return_fixture_name)

        _BENCHMARKS[f.__name__] = (f, config, fixture_params, return_fixture_func)

        return f

    if func is None:
        return decorator_benchmark
    else:
        return decorator_benchmark(func)


class AbstractBenchmark(abc.ABC):
    """
    Abstract Benchmark class that provides a basic interface for benchmarks.
    """
    output_format = None
    output_location = None
    results = None

    @abc.abstractmethod
    def run(self):
        """
        Abstract method to be overridden by subclasses to run the benchmark.
        """
        pass


class BenchmarkRunner:
    """
    Class for managing the execution of benchmarks and storage of results.
    """

    def __init__(self, benchmarks: dict):
        self.benchmarks = benchmarks

    def run_all(self):
        for benchmark_name, (benchmark, config, fixture_params, _return_fixture) in self.benchmarks.items():
            if isinstance(benchmark, AbstractBenchmark):
                benchmark.run()
                self._save_results(benchmark.results, config)
            else:
                results = benchmark()
                self._save_results(results, config)

    @staticmethod
    def _save_results(results, config):
        """
        Utility method to save benchmark results.
        This method currently supports saving results in the XML format.
        """
        if config['output_format'].lower() == 'xml':
            bu.plot_results(results, save_as=config['output_location'])
        else:
            raise ValueError(f"Unsupported output format: {config['output_format']}")


def main():
    runner = BenchmarkRunner(_BENCHMARKS)
    runner.run_all()


if __name__ == "__main__":
    main()
