import json


def read_config(config_name, dataset_name):
    with open("configs/{}.json".format(config_name), "r") as f:
        config = json.load(f)
    with open("configs/datasets/{}.json".format(dataset_name), "r") as f:
        dataset_config = json.load(f)
    assert (
        sum(dataset_config["data_percentages"]) == 100
    ), "data_percetanges do not add up to 100%"
    dataset_config["num_samples"] = [
        int(dataset_config["size"] * float(p) / 100) for p in dataset_config["data_percentages"]
    ]
    dataset_config["num_cycles"] = len(dataset_config["data_percentages"])
    config["name"] = config_name
    config["dataset"] = dataset_config
    config["dataset"]["name"] = dataset_name
    print(config)
    return config
