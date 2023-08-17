import functools
import time
import timeit
from statistics import variance
from typing import Union, Callable

import numpy as np

import utils as bu
from pydin import logging

logger = logging.stdout_color_mt(__name__)

__all__ = [
    'SingleParameter',
    'ArrayParameter',
    'GridParameter',
    'fixture',
    'analysis',
    'benchmark',
]


class SingleParameter:
    def __init__(self, value):
        self.value = value

    def values(self):
        return [self.value]


class ArrayParameter:
    def __init__(self, array):
        if isinstance(array, np.ndarray):
            self.array = array
        else:
            self.array = np.array(array)

    def values(self):
        return self.array.tolist()

    @property
    def shape(self):
        return self.array.shape


import itertools


class GridParameter:
    def __init__(self, params):
        self.params = params

    def values(self):
        grids = np.meshgrid(*[param.values() for param in self.params])
        return [grid.ravel().tolist() for grid in grids]

    def __iter__(self):
        param_values = [param.values() for param in self.params]
        return iter(itertools.product(*param_values))

    @property
    def shape(self):
        return tuple(param.shape[0] for param in self.params)


def _pydin_detail(func):
    """
    Decorator to add detailed logging, timing, and error handling to a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # logger = logging.getLogger(func.__name__)
        logger.debug(f"Starting {func.__name__} with arguments {args} and keyword arguments {kwargs}.")

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.critical(f"An error occurred in {func.__name__} : {str(e)}")
            raise
        else:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.debug(f"Finished {func.__name__} in {elapsed_time:.4f} seconds with result: {result}.")
            return result

    return wrapper


# Store registered benchmarks as a dictionary
_BENCHMARKS = {}

# Store registered fixtures
_FIXTURES = {}

# Store registered return fixtures
_ANALYSES = {}


def fixture(params=None, name=None):
    """
    Decorator to register fixture functions.
    'params' is a dictionary of parameters that can be used in the fixture function.
    'name' is a string identifier for the fixture. If not provided, the fixture function's name is used.
    """

    def decorator_fixture(f):
        # If there's no name, use function name
        nonlocal name
        if not name:
            name = f.__name__

        # Register the function and its parameters in the global _FIXTURES dictionary
        _FIXTURES[name] = (f, params)

        # The decorator returns the original function unmodified
        return f

    return decorator_fixture


@_pydin_detail
def get_benchmarks():
    """
    Returns a dictionary of registered benchmarks.
    """
    return _BENCHMARKS


class Request:
    def __init__(self):
        self._startup: Union[str, Callable] = ""
        self._execute: Union[str, Callable] = ""
        self._teardown: Union[str, Callable] = ""
        self._params = {}

    @property
    def startup(self):
        return self._format_to_string(self._startup)

    @startup.setter
    def startup(self, value):
        self._startup = value

    @property
    def execute(self):
        return self._format_to_string(self._execute)

    @execute.setter
    def execute(self, value):
        self._execute = value

    @property
    def teardown(self):
        return self._format_to_string(self._teardown)

    @teardown.setter
    def teardown(self, value):
        self._teardown = value

    @staticmethod
    def _format_to_string(value):
        if callable(value):
            # If the value is callable, recommend to pass in the source code as a string
            raise ValueError("Callable passed to Request object. Please pass the source code as a string instead.")
        return str(value)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        self._params = x


@_pydin_detail
def get_benchmark(name):
    """
    Returns a registered benchmark by name.
    """
    return _BENCHMARKS[name]


@_pydin_detail
def benchmark(func=None, *, tags=None, name=None):
    def decorator_benchmark(f):
        request = Request()
        fixture_tags = tags if tags else []
        request.params = {tag: None for tag in fixture_tags}  # Initialize with None
        f(request)

        _BENCHMARKS[name or f.__name__] = {'startup': request.startup,
                                           'execute': request.execute,
                                           'teardown': request.teardown,
                                           'tags': fixture_tags}
        return f

    if func is None:
        return decorator_benchmark
    else:
        return decorator_benchmark(func)


# @_pydin_detail
# def analysis(*args, **kwargs):
#     """
#     Decorator to register analysis functions.
#     """
#
#     def decorator(func):
#         # Add the function and any additional arguments to the registry
#         _ANALYSES[func.__name__] = (func, args, kwargs)
#
#         return func
#
#     return decorator


@_pydin_detail
def analysis(tags=None, name=None, cache=True, maxsize=None):
    """
    Decorator to register analysis functions.
    """

    def decorator_analysis(f):
        # if there's no name, use function name
        nonlocal name
        if not name:
            name = f.__name__

        # if tags is None, use name as the tag
        nonlocal tags
        if not tags:
            tags = [name]

        # # add caching if enabled
        # nonlocal cache
        # if cache:
        #     f = lru_cache(maxsize=maxsize)(f)

        # store the function and the tags in the analysis dictionary
        _ANALYSES[name] = (f, tags)
        return f

    return decorator_analysis


class BenchmarkResult:
    def __init__(self, name, loops, total_time, avg_time, variance, extra_result, params, param_type):
        self.name = name
        self.loops = loops
        self.total_time = total_time
        self.avg_time = avg_time
        self.variance = variance
        self.extra_result = extra_result
        self.params = params
        self.param_type = param_type


import os
import platform
import shelve

import hashlib
import pickle


def get_home_dir():
    if platform.system() == "Windows":
        return os.path.join(os.getenv("HOMEDRIVE"), os.getenv("HOMEPATH"))
    else:
        return os.getenv("HOME")


def filter_results(results, tags):
    """
    Filter the given benchmark results based on the specified tags.
    Returns a list of BenchmarkResult objects that match all tags.
    """
    # Ensure tags is a list to allow for multiple tags
    if isinstance(tags, str):
        tags = [tags]

    # Filter results
    filtered_results = []
    for key, res_list in results.items():
        for res in res_list:
            if any(tag == key for tag in tags):
                filtered_results.append(res)
            # if all(tag in res.tags for tag in tags):
            #     filtered_results.append(res)
    return filtered_results


def custom_autorange(timer, min_time=1.0, min_trials=10, callback=None):
    """Run timer.autorange until at least min_time has passed, or at least min_trials have been executed."""
    number = 0
    total_time = 0.0
    while total_time < min_time or number < min_trials:
        n, t = timer.autorange(callback)
        number += n
        total_time += t
    return number, total_time


class BenchmarkRunner:
    def __init__(self,
                 benchmarks=_BENCHMARKS,
                 analyses=_ANALYSES,
                 fixtures=_FIXTURES,
                 tags=None,
                 min_time=0.3,
                 min_trials=3,
                 cache=True,
                 maxsize=None):
        self.logger = logging.stdout_color_mt(__name__ + self.__class__.__name__)
        self.benchmarks = benchmarks
        self.analyses = analyses
        self.fixtures = fixtures
        self.tags = tags
        self.min_time = min_time
        self.min_trials = min_trials
        self.cache = cache
        self.maxsize = maxsize
        self.shelve_path = os.path.join(get_home_dir(), ".bor", ".cache", "benchmarks")
        os.makedirs(os.path.dirname(self.shelve_path), exist_ok=True)
        self._tag_hash_map = self.load_from_cache("_tag_hash_map", {})

    def load_from_cache(self, key, default):
        with shelve.open(self.shelve_path) as db:
            logger.debug(f"Loading from cache: {key} -> {db.get(key, default)=}")
            return db.get(key, default)

    def save_to_cache(self, key, value):
        with shelve.open(self.shelve_path) as db:
            logger.debug(f"Saving to cache: {key} -> {value}")
            db[key] = value

    def clear_cache(self):
        # os.remove(self.shelve_path)
        with shelve.open(self.shelve_path) as db:
            logger.debug(f"Clearing cache")
            db.clear()

    def run_analysis(self, analysis_name):
        analysis_func_cache = self.analyses.get(analysis_name)
        analysis_func, tags = analysis_func_cache
        logger.info(f"Running analysis: {analysis_name} -> {tags=}")
        if callable(analysis_func):
            results = self.get_results(tags)
            analysis_func(results)

    @staticmethod
    def _get_param_sets(params):
        """
        Generate a list of all possible parameter sets from the input params dictionary.
        Each item in the list is a dictionary of parameters.
        """

        # Extract keys from the parameters dictionary
        keys = params.keys()

        # Extract values from the parameters dictionary
        # If the value is an instance of SingleParameter, ArrayParameter or GridParameter,
        # call its values() method to get the list of values
        # If not, treat it as a single value parameter and wrap it in a list
        values = []
        for param in params.values():
            if isinstance(param, (SingleParameter, ArrayParameter, GridParameter)):
                values.append(param.values())
            else:
                values.append([param])

        # Generate all combinations of parameter values using itertools.product
        # Convert each combination to a dictionary using the keys extracted earlier
        param_sets = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            param_sets.append(param_dict)

        # Log the count and type of parameter sets generated
        logging.debug(f"Generated {len(param_sets)} parameter sets of type {type(param_sets)} from input parameters")

        return param_sets

    # Then, modify the get_results method to use the _tag_hash_map
    def get_results(self, tags):
        results = {}
        for tag in tags:
            results[tag] = []
            if tag in self._tag_hash_map:
                for hash in self._tag_hash_map[tag]:
                    results[tag].append(self.load_from_cache(hash, None))

        return results


    def run_all(self, benchmarks):
        for benchmark in benchmarks:
            self.run_benchmark(benchmark, self.benchmarks[benchmark])

    def run_benchmarks(self, names=None, tags=None):
        benchmarks_to_run = self.benchmarks
        if names:
            benchmarks_to_run = {
                k: v for k, v in benchmarks_to_run.items() if k in names}
        if tags:
            benchmarks_to_run = {
                k: v for k, v in benchmarks_to_run.items() if any(tag in v.get('tags', []) for tag in tags)}

        for benchmark_name, benchmark_data in benchmarks_to_run.items():
            self.run_benchmark(benchmark_name, benchmark_data)

    def _resolve_fixture_params(self, fixture_tags):
        params_dict = {}
        for tag in fixture_tags:
            fixture_func, fixture_params = self.fixtures.get(tag, (None, None))
            if callable(fixture_func):
                params_dict.update(fixture_func(fixture_params))
        return params_dict

    def run_benchmark(self, benchmark_name, benchmark_data):
        setup = benchmark_data.get('startup')
        execute = benchmark_data.get('execute')
        fixture_tags = benchmark_data.get('tags', [])
        params_dict = self._resolve_fixture_params(fixture_tags)

        param_sets = self._get_param_sets(params_dict)
        for idx, params in enumerate(param_sets):
            param_hash = hashlib.sha256(pickle.dumps((benchmark_name, params))).hexdigest()
            result = self.load_from_cache(param_hash, None)
            if result is None:
                result = self._execute_benchmark(benchmark_name, setup, execute, params, idx, fixture_tags, param_hash)

            # Update the tag-hash map and save it back to the cache
            for tag in fixture_tags:
                current_hashes = self._tag_hash_map.get(tag, [])
                if param_hash not in current_hashes:
                    current_hashes.append(param_hash)
                self._tag_hash_map[tag] = current_hashes

            self.save_to_cache("_tag_hash_map", self._tag_hash_map)
            self.logger.debug(f"Saved param set {idx + 1} for benchmark {benchmark_name} to cache.")

            self.save_to_cache(param_hash, result)

    # def _run_benchmark_with_params(self, benchmark_name, setup, execute, params, idx, fixture_tags):
    #     param_hash = hashlib.sha256(pickle.dumps((benchmark_name, params))).hexdigest()
    #     result = self.load_from_cache(param_hash, None)
    #
    #     if result is None:
    #         result = self._execute_benchmark(benchmark_name, setup, execute, params, idx, fixture_tags, param_hash)
    #
    #     # Update the tag-hash map and save it back to the cache
    #     for tag in fixture_tags:
    #         self._tag_hash_map.setdefault(tag, []).append(param_hash)
    #
    #     self.save_to_cache("_tag_hash_map", self._tag_hash_map)
    #     self.logger.debug(f"Saved param set {idx + 1} for benchmark {benchmark_name} to cache.")

    def _execute_benchmark(self, benchmark_name, setup, execute, params, idx, fixture_tags, param_hash):
        globals_ = {'__name__': '__main__', 'params': params}
        self.logger.debug(
            f"Running "
            f"benchmark: {benchmark_name}, "
            f"setup: {setup}, "
            f"execute: {execute}, "
            f"globals: {globals_}".replace("\n", ""))

        timer = timeit.Timer(execute, setup=setup, globals=globals_)
        times = []
        loops, total_time = custom_autorange(timer,
                                             callback=lambda n, t: times.append(t / n),
                                             min_time=self.min_time,
                                             min_trials=self.min_trials)

        # Calculate variance
        time_variance = variance(times)
        print(times)
        # Create a BenchmarkResult object
        result = BenchmarkResult(
            name=benchmark_name,
            loops=loops,
            total_time=total_time,
            avg_time=sum(times) / len(times),
            variance=time_variance,
            extra_result=None,  # Placeholder until you decide what extra results to include
            params=params,  # Added parameter details
            param_type=type(params).__name__,  # Added parameter type
        )

        self.logger.debug(
            f"Finished running parameter set {idx + 1} for benchmark {benchmark_name}")

        self.logger.info(
            f"Benchmark: {benchmark_name}, "
            f"Loops: {loops}, "
            f"Total time: {self.format_time(total_time)}, "
            f"Average execution time: {self.format_time(sum(times) / len(times))}, "
            f"Variance: {self.format_time(time_variance)}")

        return result

    @staticmethod
    def format_time(time_seconds):
        if time_seconds >= 1:
            return f"{time_seconds:.3f} s"
        elif time_seconds >= 0.001:
            return f"{time_seconds * 1e3:.3f} ms"
        elif time_seconds >= 0.000001:
            return f"{time_seconds * 1e6:.3f} µs"
        else:
            return f"{time_seconds * 1e9:.3f} ns"

    def _run_benchmarks_by_tag(self, tag):
        self.logger.info(f"Running benchmarks with tag: {tag}")
        for benchmark_name, benchmark_data in self.benchmarks.items():
            if tag in benchmark_data.get('tags', []):
                self._run_single_benchmark(benchmark_name)


def main():
    runner = BenchmarkRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
