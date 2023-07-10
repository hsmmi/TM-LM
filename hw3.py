# Read dataset
from my_io import my_io
from preproccessing import preproccessing

dataset_path = "./Dataset/"
dataset = my_io(dataset_path).read_jsons_from_folder()

# Preproccessing
tokens = preproccessing().tokenize(dataset)

print("H")
