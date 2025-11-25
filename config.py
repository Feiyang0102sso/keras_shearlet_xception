# config.py
import logging
import sys
from pathlib import Path

# --- root path ---
ROOT_DIR = Path(__file__).parent.resolve()

# --- directory paths ---
DATA_PATH_NAME = 'data'
DATA_PATH = ROOT_DIR / DATA_PATH_NAME

RESULT_PATH_NAME = 'result'
RESULT_PATH = ROOT_DIR / RESULT_PATH_NAME

RESULT_FAST_PATH_NAME = 'result_fast'
RESULT_FAST_PATH = ROOT_DIR / RESULT_FAST_PATH_NAME

# --- directory helper methods ---
def ensure_dir(path: Path):
    """If the path does not exist, create it."""
    path.mkdir(parents=True, exist_ok=True)

def _get_dir(dir_path: Path, create: bool = False) -> Path:
    """
    An internal helper function used to retrieve the directory path.
    If create=True, the directory will be created if it does not exist.
    Otherwise, an exception will be thrown if the directory does not exist.
    """
    if create:
        dir_path.mkdir(parents=True, exist_ok=True)
    elif not dir_path.exists():
        raise FileNotFoundError(f"Directory '{dir_path}' not exist")
    return dir_path

def get_data_dir() -> Path:
    return _get_dir(DATA_PATH)

def get_result_dir() -> Path:
    return _get_dir(RESULT_PATH, create=True)



# --- logging configs ---

# logging colors
COLOR_DEBUG = "\033[36m"  # 青色
COLOR_INFO = "\033[32m"  # 绿色
COLOR_WARNING = "\033[33m"  # 黄色
COLOR_ERROR = "\033[31m"  # 红色
COLOR_CRITICAL = "\033[41m"  # 红底
RESET = "\033[0m"


def setup_logger(name: str = "Shearlet_Xception"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.propagate = False

    # stdout: DEBUG, INFO, WARNING
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    def stdout_format(record):
        if record.levelno == logging.DEBUG:
            color = COLOR_DEBUG
        elif record.levelno == logging.INFO:
            color = COLOR_INFO
        elif record.levelno == logging.WARNING:
            color = COLOR_WARNING
        else:
            color = ""
        return f"{color}[{record.levelname}] {record.getMessage()}{RESET}"

    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    stdout_handler.format = stdout_format

    # stderr: ERROR, CRITICAL
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)

    def stderr_format(record):
        if record.levelno == logging.ERROR:
            color = COLOR_ERROR
        elif record.levelno == logging.CRITICAL:
            color = COLOR_CRITICAL
        else:
            color = ""
        return (f"{color}[{record.levelname}] "
                f"[{record.name} - {record.filename}:{record.lineno}] "
                f"{record.getMessage()}{RESET}")

    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    stderr_handler.format = stderr_format

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


# global logger，other module use 'from config import logger'
logger = setup_logger()

# add a function for logging into files
def add_file_handler(log_path_str: str, mode: str = 'w'):
    """
    :param log_path_str:
        1. If only a filename is passed (e.g., "train.log") -> it will be stored in the default RESULT_PATH.
        2. If a relative path is passed (e.g., "experiments/exp1/train.log") -> a folder will be created and stored in the project root directory or the current directory.
        3. If an absolute path is passed (e.g., "/tmp/logs/train.log") -> the absolute path will be used.
    :param mode: 'w' overwrite, 'a' append
        """

    path_obj = Path(log_path_str)

    if len(path_obj.parts) == 1:
        final_log_path = get_result_dir() / path_obj
    else:
        final_log_path = path_obj.resolve()

    final_log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(final_log_path, mode=mode, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    file_fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_fmt)

    logger.addHandler(file_handler)

    print(f"{COLOR_INFO}Log file saved at: {final_log_path}{RESET}")

# --- example logs ---
# logger.debug("debug")
# logger.info("info")
# logger.warning("warning")
# logger.error("error")
# logger.critical("critical")