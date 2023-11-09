import os
from glob import glob
import shutil

log_dir = "/home/RT_Paper/temp"
date_list = os.listdir(log_dir)
for date in date_list:
    result_file = f"{log_dir}/{date}/result/test_result.csv"
    if not os.path.isfile(result_file):
        shutil.rmtree(f"{log_dir}/{date}")
