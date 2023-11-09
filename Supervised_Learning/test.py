import json
import os
from glob import glob
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# cpu only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from natsort import natsorted
from modules import model as model
from modules import supervised as supervised


def main():
    JSON = "/home/RT_Paper/log/202310180125/config.json"
    DATE = JSON.split("/")[-2]
    config = json.load(open(JSON))
    model_list = "/".join(JSON.split("/")[0:-1]) + "/models"
    model_list = natsorted(glob(model_list + "/*.h5", recursive=True))
    model_path = model_list[0]
    print(model_path)
    config["MODEL_PATH"] = model_path
    config["MODE"] = "test"
    config["TEST_DATA_PATH"] = "/home/RT_Paper/data/Classification/Exp3/test"
    config["DATE"] = DATE
    config["REPREDICT"] = "Y"
    config["THRESHOLD"] = 0.3
    config["SAVE_FALSE_IMAGE"] = "Y"
    Experiment = supervised.Experiment_Model(**config)
    Experiment.test_model()
    Experiment.report_model()
    Experiment.upload_notion()

    del Experiment
    del config


if __name__ == "__main__":
    main()
