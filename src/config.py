import glob
import logging
import sys
import os
from datetime import date
import random
import string
from typing import Callable

CHAR_POOL = string.ascii_lowercase + string.digits


def random_str(length: int = 6, conflicts: list = []):
    while ret := ''.join(random.choice(CHAR_POOL) for _ in range(length)):
        if ret not in conflicts:
            break
    return ret


class Config():
    LOG_ROOT = 'log'

    DEFAULT_LEVEL = logging.INFO
    DEFAULT_LOG_FORMAT = '%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    DEFAULT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def _set_stdout(verbose: bool = True,
                    prompt: str = 'log setup OK -> stdout'):
        logging.basicConfig(stream=sys.stdout,
                            level=Config.DEFAULT_LEVEL,
                            format=Config.DEFAULT_LOG_FORMAT,
                            datefmt=Config.DEFAULT_TIME_FORMAT)
        if verbose:
            logging.info(f'{prompt}')

    def _set_logger(log_dir: str,
                    log_name: str,
                    verbose: bool = True,
                    prompt: str = 'log setup OK -> logger:'):
        os.makedirs(log_dir, exist_ok=True)
        log_path = f'{log_dir}/{log_name}.log'
        logging.basicConfig(filename=log_path,
                            level=Config.DEFAULT_LEVEL,
                            format=Config.DEFAULT_LOG_FORMAT,
                            datefmt=Config.DEFAULT_TIME_FORMAT)
        if verbose:
            log_abspath = os.path.abspath(log_path).replace('\\', '/')
            print(f'file:///{log_abspath}')
            logging.info(f'{prompt} {log_abspath}')

    def __set(self, stdout: bool, dry_run: bool, prompt: str):
        if stdout:
            Config._set_stdout(verbose=self.verbose)
        else:
            Config._set_logger(self.log_dir,
                               self.log_name,
                               verbose=self.verbose)
        self.dry_run = dry_run
        if self.verbose:
            logging.info(f'{prompt} [stdout={stdout}], [dry_run={dry_run}]')

    def __init__(self, stdout: bool, dry_run=False, verbose=True, exp_code: str = None) -> None:
        self.verbose = verbose
        log_base = f'{Config.LOG_ROOT}/{date.today()}'
        if exp_code is None:
            while code := random_str():
                if not os.path.exists(f'{log_base}/{code}'):
                    break
        else:
            code = exp_code

        self.exp_code = code
        self.log_dir = f'{log_base}/{code}'
        self.log_name = code
        self.__set(stdout, dry_run, f'experiment "{code}"')

    def reset(self, stdout: bool, dry_run: bool) -> None:
        self.__set(stdout, dry_run, 'reset complete')

    def save(self, fn: Callable, *args, **kwargs) -> None:
        if self.dry_run:
            return
        fn(self.log_dir, *args, **kwargs)

    def save_fp(self, fn: Callable, filename: str, *args, **kwargs) -> None:
        if self.dry_run:
            return
        with open(os.path.join(self.log_dir, filename), 'w') as fp:
            fn(fp, *args, **kwargs)
