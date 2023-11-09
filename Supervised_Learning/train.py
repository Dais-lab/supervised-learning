import json
import os
from glob import glob
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 0번 GPU만 사용하도록 설정

from natsort import natsorted
from modules import supervised as supervised


def main():
    JSON_list = natsorted(
        glob("/home/RT_Paper/src/Supervised_Learning/json_list/*.json")
    )

    for JSON in JSON_list:
        config = json.load(open(JSON))
        print(JSON)
        Experiment = supervised.Experiment_Model(**config)
        Experiment.save_config(JSON)
        Experiment.train_model()
        Experiment.test_model()
        Experiment.report_model()
        Experiment.upload_notion()

        del Experiment
        del config
        os.remove(JSON)


if __name__ == "__main__":
    main()
