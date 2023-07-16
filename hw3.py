# Read dataset
from my_io import my_io
from preproccessing import preproccessing
from LM import LM

dataset_path = "./Dataset/"
dataset = my_io(dataset_path).read_jsons_from_folder()

# Preproccessing
tokens = preproccessing().tokenize(dataset)

remove_stop_word = preproccessing().remove_stop_word(tokens)

normalization = preproccessing().normalization(remove_stop_word)

stemming = preproccessing().stemming(normalization)

remove_stop_word = preproccessing().remove_stop_word(stemming)

# Save preproccessed dataset
dataset = remove_stop_word

# Language Model
unigram_model = LM(dataset).estimate_unigram()
bigram_model = LM(dataset).estimate_bigram()

# Find top k words most likely to follow a given word in the Bigram language
# model with Linear Interpolation Smoothing
word = "decent"
k = 10
lambda_value = 0.9
top_words = LM(dataset).find_top_words(word, k, lambda_value)
