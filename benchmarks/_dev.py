
# def generate_benchmark(func_name, module_name, extra_params=None):
#     if extra_params is None:
#         extra_params = {}
#
#     @bench.benchmark(tags=['linspace'], name=f'pydin.{func_name}')
#     def benchmark_pydin_linspace(request):
#         request.startup = textwrap.dedent(f"""
#         import pydin
#         """)
#         request.execute = textwrap.dedent(f"""
#         pydin.core.linalg.{func_name}(**params, **extra_params)
#         """)
#
#     benchmark_pydin_linspace.__name__ = f'benchmark_pydin_{func_name}'
#     benchmark_pydin_linspace.__qualname__ = f'benchmark_pydin_{func_name}'
#     benchmark_pydin_linspace.__module__ = module_name
#     benchmark_pydin_linspace.__doc__ = f"""
#     Benchmark {func_name} in {module_name}
#     """
#
#     return benchmark_pydin_linspace
# @bench.fixture(name='meshgrid', params=MESHGRID_PARAMS)
# def meshgrid_fixture(params, **kwargs):
#     return params

# @bench.benchmark(tags=['linspace'], name='pydin.linspace', function='linspace', module='pydin.core.linalg',
#                  extra_params={'parallel': [True, False]})
# @bench.benchmark(tags=['linspace'], name='pydin.eigen.linspace', function='linspace', module='pydin.core.linalg.eigen')
# @bench.benchmark(tags=['logspace'], name='numpy.logspace', function='logspace', module='numpy')
# @bench.benchmark(tags=['logspace'], name='pydin.logspace', function='logspace', module='pydin.core.linalg',
#                  extra_params={'parallel': [True, False]})
# @bench.benchmark(tags=['geomspace'], name='numpy.geomspace', function='geomspace', module='numpy')
# @bench.benchmark(tags=['geomspace'], name='pydin.geomspace', function='geomspace', module='pydin.core.linalg',
#                  extra_params={'parallel': [True, False]})
# @bench.benchmark(tags=['meshgrid'], name='numpy.meshgrid', function='meshgrid', module='numpy')
# @bench.benchmark(tags=['meshgrid'], name='pydin.meshgrid', function='meshgrid', module='pydin.core.linalg',
#                  extra_params={'parallel': [True, False]})
# @bench.benchmark(tags=['meshgrid'], name='pydin.eigen.meshgrid', function='meshgrid', module='pydin.core.linalg.eigen')
# def generic_benchmark(request):
#     import_str = f"import {request.module} as mod"
#     exec(import_str)
#
#     func_str = f"mod.{request.function}(**request.fixture_params)"
#     if 'extra_params' in request:
#         for key, values in request.extra_params.items():
#             for value in values:
#                 new_params = request.fixture_params.copy()
#                 new_params.update({key: value})
#                 func_str = f"mod.{request.function}(**new_params)"
#
#     request.startup = lambda: exec(import_str)
#     request.execute = lambda: exec(func_str)
#     request.teardown = lambda: exec("del mod")
