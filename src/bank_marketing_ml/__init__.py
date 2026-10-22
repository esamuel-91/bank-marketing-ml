import logging as lg
from datetime import datetime
from from_root import from_root

import os
import sys

LOG_DIR = from_root("logs", "running_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

## creating logger
logger = lg.getLogger(__name__)
logger.setLevel(lg.INFO)

## creating handler and its formatter
filehandler = lg.FileHandler(LOG_FILE_PATH)
sthandler = lg.StreamHandler(sys.stdout)
formatter = lg.Formatter("[%(asctime)s] %(levelname)s - %(module)s: %(message)s")
filehandler.setFormatter(formatter)
sthandler.setFormatter(formatter)

## adding handler to the logger
logger.addHandler(filehandler)
logger.addHandler(sthandler)
