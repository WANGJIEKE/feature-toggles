import ast
import astor
from collections import defaultdict
import functools
import itertools
import os
import pandas as pd
import sys
from typing import Dict, Iterable, List, Set, Tuple, Union


# import Path from correct library based on Python version;
# use builtin pathlib if available; otherwise use the backport version
if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
    from pathlib import Path  # pylint: disable=import-error
else:
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
            yield from walk_dir_for_ext(path, ext)
        if path.is_file() and path.suffix == ext:  # avoid broken symlink
            yield path


def get_lib_name_from_repo_name(repo_name):
    # type: (str) -> str
    """repo_name is in the format of library_user#repo"""
    lib, _ = repo_name.split('_', maxsplit=1)
    return lib


def get_rel_path(p, root):
    # type: (Path, Path) -> str
    """get relative path for p based on root
    e.g. if p is "/a/b/c" and root is "/a", then return "b/c"
    """
    return os.path.relpath(str(p), start=str(root))


def find_func_usage_naive(path, root, funcs):
    # type: (Path, Path, Iterable[str]) -> Dict[str, List[Tuple[str, int str]]]
    """iterate through file and find all appearance of function name in funcs
    the structure of returned dict is {'func_name': [('file_rel_path', line_number, line_content), ...]}
    """
    d = defaultdict(list)
    n = 1
    rel_path = get_rel_path(path, root)
    try:
        with path.open() as f:
            for line in f:
                for func in funcs:
                    if func in line:
                        d[func].append((rel_path, n, line.rstrip('\n')))
                n += 1
    except Exception as e:
        print(f'Exception with type {type(e)} caught when reading file {rel_path}: {e}', file=sys.stderr)
    return d


def find_func_usage_ast(path, root, funcs):
    # type: (Path, Path, Iterable[str]) -> Dict[str, List[Tuple[str, int str]]]
    """build the ast from the file, and find all ast.Call nodes; same returned structure as above function
    """
    d = defaultdict(list)
    rel_path = get_rel_path(path, root)
    try:
        with path.open() as f:
            source = f.read()
            tree = ast.parse(source, filename=rel_path)
            call_nodes = filter(lambda n: type(n) is ast.Call, ast.walk(tree))

            for call in call_nodes:
                if type(call.func) is ast.Name:
                    if call.func.id in funcs:
                        d[call.func.id].append((rel_path, call.lineno, repr(astor.to_source(call))))
                elif type(call.func) is ast.Attribute:
                    if call.func.attr in funcs:
                        d[call.func.attr].append((rel_path, call.lineno, repr(astor.to_source(call))))
                else:
                    print(f'unexpected type from call.func={call.func}', file=sys.stderr)
                    print(f'\tsource={astor.to_source(call)}', file=sys.stderr)
                    print(f'\tpath={rel_path}', file=sys.stderr)
                    print(f'\tlineno={call.lineno}', file=sys.stderr)
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
    config = None

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

        # usage = functools.reduce(
        #     merge_func_usage,
        #     map(lambda py_file: find_func_usage_naive(py_file, repo_dir, funcs), py_files),
        #     {}
        # )

        usage_ast = functools.reduce(
            merge_func_usage,
            map(lambda py_file: find_func_usage_ast(py_file, repo_dir, funcs), py_files),
            {}
        )

        # func_usage_dict_to_df(usage).to_csv(repo_stats_root / (repo_dir.stem + '.csv'), index=False)
        func_usage_dict_to_df(usage_ast).to_csv(repo_stats_root / (repo_dir.stem + '.ast.csv'), index=False)
