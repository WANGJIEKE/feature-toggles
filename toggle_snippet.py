# -*- coding: future_fstrings -*-
from __future__ import print_function
import ast
import astor
from collections import defaultdict
import functools
import itertools
import os
import pandas as pd
import sys
from typing import Dict, Iterable, List, Set, Tuple, Union


# import Path from pathlib
# use the backport version if there is no builtin in current Python version
try:
    from pathlib import Path  # pylint: disable=import-error
except ImportError:
    # https://pypi.org/project/pathlib2/
    from pathlib2 import Path  # pylint: disable=import-error


def load_all_used_func(f):
    # type: (Path) -> pd.DataFrame
    """load toggle library function list sheet"""
    return pd.read_excel(f, sheet_name=1).dropna(how='all')


def parse_lib_funcs(df):
    # type: (pd.DataFrame) -> Dict[str, Set[str]]
    """convert toggle library function list to dictionary
    the structure of returned dict is {'library1': {'func1', 'func2', ...}, 'library2': ...}
    """
    d = defaultdict(set)
    for _, row in df.iterrows():
        d[row['Library']].add(row['used function'])
    return d


def walk_dir_for_ext(root, ext):
    # type: (Path, str) -> Iterable[Path]
    """walk through the directory starting from root and yield all file path having an extension of ext"""
    for path in root.iterdir():
        if path.is_dir():
            for file_in_sub_path in walk_dir_for_ext(path, ext):
                yield file_in_sub_path
        if path.is_file() and path.suffix == ext:  # avoid broken symlink
            yield path


def get_lib_name_from_repo_name(repo_name):
    # type: (str) -> str
    """repo_name is in the format of library_user#repo"""
    lib, _ = repo_name.split('_', 1)
    return lib


def get_rel_path(p, root):
    # type: (Path, Path) -> str
    """get relative path for p based on root
    e.g. if p is "/a/b/c" and root is "/a", then return "b/c"
    """
    return os.path.relpath(str(p), start=str(root))


def find_func_usage_ast(path, root, toggles):
    # type: (Path, Path, Iterable[str]) -> Dict[str, List[Tuple[str, int, str]]]
    """build the ast from the file, and find all ast.Call nodes
    the structure of returned dict is {'toggle_name': [('file_rel_path', line_number, line_content), ...]}
    """
    d = defaultdict(list)
    rel_path = get_rel_path(path, root)
    try:
        with path.open() as f:
            source = f.read()
            tree = ast.parse(source, filename=rel_path)
            toggle_nodes = filter(lambda n: isinstance(n, (ast.Call, ast.ClassDef)), ast.walk(tree))

            for node in toggle_nodes:
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in toggles:
                            d[node.func.id].append((rel_path, node.lineno, repr(astor.to_source(node))))
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in toggles:
                            d[node.func.attr].append((rel_path, node.lineno, repr(astor.to_source(node))))
                    else:
                        print(f'unexpected type from call.func={node.func}', file=sys.stderr)
                        print(f'\tsource={astor.to_source(node)}', file=sys.stderr)
                        print(f'\tpath={rel_path}', file=sys.stderr)
                        print(f'\tlineno={node.lineno}', file=sys.stderr)
                else:  # isinstance(node, ast.ClassDef)
                    for base_class in node.bases:
                        if isinstance(base_class, ast.Name):
                            if base_class.id in toggles:
                                d[base_class.id].append((rel_path, node.lineno, repr(astor.to_source(node))))
                        elif isinstance(base_class, ast.Attribute):
                            if base_class.attr in toggles:
                                d[base_class.attr].append((rel_path, node.lineno, repr(astor.to_source(node))))
                        else:
                            print(f'unexpected type from call.func={base_class}', file=sys.stderr)
                            print(f'\tsource={astor.to_source(node)}', file=sys.stderr)
                            print(f'\tpath={rel_path}', file=sys.stderr)
                            print(f'\tlineno={node.lineno}', file=sys.stderr)

    except Exception as e:
        print(f'Exception with type {type(e)} caught when reading file {rel_path}: {e}', file=sys.stderr)
    return d


def merge_func_usage(d1, d2):
    # type: (Dict[str, List[Tuple[str, int, str]]], Dict[str, List[Tuple[str, int, str]]]) -> Dict[str, List[Tuple[str, int, str]]]
    """merge the two usage dicts {'func1': [('file1_rel_path', line_number1, line_content1), ...]} together"""
    if len(d1) == 0:
        return d2
    if len(d2) == 0:
        return d1
    d = defaultdict(list)
    for func_name, func_usage in itertools.chain(d1.items(), d2.items()):
        d[func_name].extend(func_usage)
    return d


def func_usage_dict_to_df(d):
    # type: (Dict[str, List[Tuple[str, int, str]]]) -> pd.DataFrame
    """convert usage dict {func_name: [('file1_rel_path', line_number1, line_content1), ...]}
    to pandas.DataFrame
    """
    table = {'func': [], 'path': [], 'line_number': [], 'line_content': []}
    for func_name, usage in d.items():
        table['func'].extend(func_name for _ in range(len(usage)))
        for p, line_num, line_content in usage:
            table['path'].append(p)
            table['line_number'].append(line_num)
            table['line_content'].append(line_content)
    return pd.DataFrame(table)


if __name__ == '__main__':
    import json

    with Path('./settings.json').open() as config_file:
        config = json.load(config_file)

    repo_root = Path(config['repo_root'])
    repo_stats_root = Path(config['repo_stats_root'])
    repo_stats_root.mkdir(exist_ok=True)
    excel_file = Path(config['excel_file'])

    all_used_func = load_all_used_func(excel_file)
    lib_funcs = parse_lib_funcs(all_used_func)

    for repo_dir in repo_root.iterdir():
        if not repo_dir.is_dir():  # F to .DS_Store
            continue
        print(f'searching through repo {repo_dir}...', file=sys.stderr)
        funcs = lib_funcs[get_lib_name_from_repo_name(repo_dir.stem)]
        py_files = [py_file for py_file in walk_dir_for_ext(repo_dir, '.py')]

        usage_ast = functools.reduce(
            merge_func_usage,
            map(lambda py_file: find_func_usage_ast(py_file, repo_dir, funcs), py_files),
            {}
        )

        func_usage_dict_to_df(usage_ast).to_csv(repo_stats_root / (repo_dir.stem + '.ast.csv'), index=False)
