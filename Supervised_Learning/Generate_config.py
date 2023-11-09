import json
import os
from pprint import pprint
import shutil


def main():
    shutil.rmtree("/home/RT_Paper/src/Supervised_Learning/json_list")
    os.makedirs("/home/RT_Paper/src/Supervised_Learning/json_list")
    base_config = json.load(open("/home/RT_Paper/src/Supervised_Learning/Sample.json"))
    base_config["DATE"] = ""
    base_config["MEMO"] = ""
    base_config["MODE"] = "train"
    base_config["OPTIMIZER"] = "Adam"
    base_config["MODEL_PATH"] = ""
    base_config["TARGET_SIZE"] = [512, 512, 1]
    base_config["EPOCHS"] = 9999
    base_config["LOG_DIR"] = "/home/RT_Paper/log"
    base_config["NOTION_DATABASE_ID_FS"] = "1f959176342148768bff07bacec5c0b7"
    base_config["NOTION_KEY"] = "secret_2nVjIaYGdiJJbz7VpKwF0kqsdbZyqgjgLHPQWfyEXzF"
    base_config["TENSORBOARD_LOG_DIR"] = "/data/tensorboard_log"
    base_config["REPREDICT"] = "N"
    base_config["THRESHOLD"] = 0.3
    base_config["SAVE_FALSE_IMAGE"] = "y"
    base_config["VAL_DATA_PATH"] = ""

    model_type = ["VGG16"]
    data_type = {"Vanilla-PO": "Exp1"}  # ,
    loss = ["binary_focal_crossentropy"]
    class_weight = [{"0": 1, "1": 643}, {"0": 1, "1": 3}]
    batch_size = [32, 64]
    learning_rate = [0.0003, 0.0005, 0.0007]
    patience = [20]
    for model in model_type:
        for batch in batch_size:
            for lr in learning_rate:
                for pat in patience:
                    for data in data_type:
                        if data == "Vanilla-PO":
                            for los in loss:
                                config = base_config.copy()
                                config["DATA_TYPE"] = data
                                config["MODEL_TYPE"] = model
                                config["BATCH_SIZE"] = batch
                                config[
                                    "TRAIN_DATA_PATH"
                                ] = f"/home/RT_Paper/data/Classification/{data_type[data]}/train"
                                config[
                                    "TEST_DATA_PATH"
                                ] = f"/home/RT_Paper/data/Classification/{data_type[data]}/test"
                                config["LEARNING_RATE"] = lr
                                config["PATIENCE"] = pat
                                config["LOSS_FUNC"] = los
                                config["WEIGHT"] = (
                                    class_weight[0]
                                    if los == "binary_crossentropy"
                                    else class_weight[1]
                                )
                                pprint(config)
                                json.dump(
                                    config,
                                    open(
                                        "/home/RT_Paper/src/Supervised_Learning/json_list/{}_{}_{}_{}_{}_{}.json".format(
                                            data, model, batch, lr, pat, los
                                        ),
                                        "w",
                                    ),
                                    indent=4,
                                )
                                del config

                        else:
                            config = base_config.copy()
                            config["DATA_TYPE"] = data
                            config["MODEL_TYPE"] = model
                            config["BATCH_SIZE"] = batch
                            config[
                                "TRAIN_DATA_PATH"
                            ] = f"/home/RT_Paper/data/Classification/{data_type[data]}/train"
                            config[
                                "TEST_DATA_PATH"
                            ] = f"/home/RT_Paper/data/Classification/{data_type[data]}/test"
                            config["LEARNING_RATE"] = lr
                            config["PATIENCE"] = pat
                            config["LOSS_FUNC"] = "binary_crossentropy"
                            config["WEIGHT"] = {"0": 1, "1": 3}
                            pprint(config)
                            json.dump(
                                config,
                                open(
                                    "/home/RT_Paper/src/Supervised_Learning/json_list/{}_{}_{}_{}_{}.json".format(
                                        data, model, batch, lr, pat
                                    ),
                                    "w",
                                ),
                                indent=4,
                            )
                            del config


if __name__ == "__main__":
    main()
