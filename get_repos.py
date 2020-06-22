import os
import pandas as pd
from pathlib import Path
import random
import time
from urllib.parse import urlparse


repo_root = Path('../repos')
excel_file = Path('../toggle(3).xlsx').resolve(strict=True)

if not repo_root.exists():
    repo_root.mkdir()
repo_root = repo_root.resolve()

repo_list_df = pd.read_excel(excel_file)
url_list = repo_list_df['URL'].tolist()
lib_list = repo_list_df['Toggle Library'].tolist()

with Path('../get_repo_log.log').open('w') as log_file:
    for url, lib in random.sample(list(zip(url_list, lib_list)), 5):
        p_url = Path(urlparse(url).path)
        save_dir_name = f'{lib}_{p_url.parent.stem}#{p_url.stem}'
        cmd = f'git clone {url} "{repo_root / save_dir_name}"'
        print(cmd)
        if os.system(cmd) != 0:
            log_file.write(f'"{cmd}" failed')
        time.sleep(random.randint(1, 3))
