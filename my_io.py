# Read dataset which is in JSON format
import os
import json
import pickle


dataset_path = "./Dataset/"


class my_io:
    def __init__(self, dataset_folder_path=None):
        if dataset_folder_path is not None:
            self.dataset_folder_path = dataset_folder_path
        self.data = []

    def read_jsons_from_folder(self, dataset_folder_path=None):
        if dataset_folder_path is not None:
            self.dataset_folder_path = dataset_folder_path
        data = []
        for filename in os.listdir(self.dataset_folder_path):
            if filename.endswith(".json"):
                with open(self.dataset_folder_path + filename) as f:
                    json_file = json.load(f)

                    reviews = json_file["Reviews"]
                    obj = []
                    for review in reviews:
                        obj.append(review["Content"])

                    data.append(obj)
        self.data = data
        return data

    def save_data(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
