# Read dataset
from my_io import my_io
from preproccessing import preproccessing
from LM import LM

dataset = my_io().load_data("preproccessed_dataset.json")
unigram_model = my_io().load_data("unigram_model.pkl")
bigram_model = my_io().load_data("bigram_model.pkl")


# Find top k words most likely to follow a given word in the Bigram language
# model with Linear Interpolation Smoothing
word = "decent"
k = 10
lambda_value = 0.9
top_words = LM(dataset).find_top_words(word, k, lambda_value)

print(f"Top {k} words most likely to follow {word} are:")
print(top_words)
