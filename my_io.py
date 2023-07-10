# Read dataset which is in JSON format
import os
import json


dataset_path = "./Dataset/"


class my_io:
    def __init__(self, dataset_folder_path):
        self.dataset_folder_path = dataset_folder_path
        self.data = []

    def read_jsons_from_folder(self):
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


# Read dataset
# data = my_io(dataset_path).read_jsons_from_folder()

# print("H")
