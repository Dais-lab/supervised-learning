import json
import os
from glob import glob
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from natsort import natsorted
from modules import model as model
from modules import supervised as supervised


# 모~~든 모델을 다시 테스트 하는 코드


def main():
    JSON_list = natsorted(glob("/home/RT_Paper/log/**/*.json", recursive=True))

    for JSON in JSON_list:
        print(JSON)
        DATE = JSON.split("/")[-2]
        config = json.load(open(JSON))
        model_list = "/".join(JSON.split("/")[0:-1]) + "/models"
        # model_list = natsorted(glob(model_list + "/*.h5", recursive=True))
        # model_path = model_list[0]
        config["MODEL_PATH"] = "model_path"
        config["MODE"] = "test"
        config["TEST_DATA_PATH"] = "/home/RT_Paper/data/Classification실험/Exp3/test"
        config["DATE"] = DATE
        config["REPREDICT"] = "N"
        config["THRESHOLD"] = 0.3
        config["SAVE_FALSE_IMAGE"] = "Y"
        config["VAL_DATA_PATH"] = ""
        config["NOTION_DATABASE_ID_FS"] = "2c5d92fd1ef441c3a4443830a2a35ad4"
        config["LOG_DIR"] = "/home/RT_Paper/log"
        Experiment = supervised.Experiment_Model(**config)
        # Experiment.save_config(JSON)
        # Experiment.train_model()
        Experiment.test_model()
        Experiment.report_model()
        Experiment.upload_notion()

        del Experiment
        del config


if __name__ == "__main__":
    main()
