import pybind11_stubgen
import re
import argparse
import os
import subprocess


def process_templates(match: re.Match):
    template = match.group(0)
    return template.replace('<', '[').replace('>', ']')


def strip_templates_from_docstrings(docstring: str):
    template_pattern = re.compile(r'\w+<\w+>')
    return template_pattern.sub(process_templates, docstring)


if __name__ == '__main__':
    pybind11_stubgen.function_docstring_preprocessing_hooks.append(strip_templates_from_docstrings)

    # pybind11_stubgen.function_docstring_preprocessing_hooks.append(
    #     strip_dimension_from_std_array
    # )

    pybind11_stubgen.main()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('module_name', type=str, help='Name of the module to generate stubs for')
    # parser.add_argument('-o', '--output_dir', type=str, help='Output directory for generated stubs')
    # parser.add_argument('--formatter_path', type=str, help='Path to the formatter executable')
    #
    # args = parser.parse_args()
    #
    # pybind11_stubgen.main(module_names=[args.module_name], output_dir=args.output_dir)
    #
    # # Run code formatter on generated stubs
    # if args.formatter_path and os.path.exists(args.formatter_path):
    #     subprocess.run([args.formatter_path, args.output_dir])
