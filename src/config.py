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


class Config:
    LOG_ROOT = 'log'

    DEFAULT_LEVEL = logging.INFO
    DEFAULT_LOG_FORMAT = '%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    DEFAULT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    @staticmethod
    def _set_stdout(verbose: bool = True,
                    prompt: str = 'log setup OK -> stdout'):
        logging.basicConfig(stream=sys.stdout,
                            level=Config.DEFAULT_LEVEL,
                            format=Config.DEFAULT_LOG_FORMAT,
                            datefmt=Config.DEFAULT_TIME_FORMAT)
        if verbose:
            logging.info(f'{prompt}')

    @staticmethod
    def _set_logger(log_dir: str,
                    verbose: bool = True,
                    prompt: str = 'log setup OK -> logger:'):
        os.makedirs(log_dir, exist_ok=True)
        log_path = f'{log_dir}/.log'
        logging.basicConfig(filename=log_path,
                            level=Config.DEFAULT_LEVEL,
                            format=Config.DEFAULT_LOG_FORMAT,
                            datefmt=Config.DEFAULT_TIME_FORMAT)
        if verbose:
            log_abspath = os.path.abspath(log_path).replace('\\', '/')
            print(f'file:///{log_abspath}')
            logging.info(f'{prompt} {log_abspath}')

    @staticmethod
    def log(stdout: bool, level: int, msg: str, log_dir: str = None):
        Config.__set(stdout, False, log_dir)
        logging.log(level, msg)

    @staticmethod
    def __set(stdout: bool, verbose: bool, log_dir: str = None):
        if stdout:
            Config._set_stdout(verbose=verbose)
        else:
            Config._set_logger(log_dir, verbose=verbose)

    def __init__(self,
                 stdout: bool,
                 dry_run=False,
                 verbose=True,
                 exp_code: str = random_str()) -> None:
        self.stdout = stdout
        self.verbose = verbose
        log_base = f'{Config.LOG_ROOT}/{date.today()}'
        while os.path.exists(f'{log_base}/{exp_code}'):
            exp_code = random_str()

        self.exp_code = exp_code
        self.log_dir = f'{log_base}/{exp_code}'
        self.dry_run = dry_run
        self.__set(stdout, verbose, self.log_dir)
        if verbose:
            logging.info(f'experiment "{exp_code}" [stdout={stdout}], [dry_run={dry_run}]')

    def clone(self) -> 'Config':
        return Config(self.stdout, self.dry_run, self.verbose, self.exp_code)

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
