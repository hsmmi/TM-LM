import nltk
from nltk.util import ngrams
from collections import Counter


# Function to estimate Unigram language model using MLE
def estimate_unigram(tokens):
    unigram_counts = Counter(tokens)
    total_tokens = len(tokens)
    unigram_model = {
        token: count / total_tokens for token, count in unigram_counts.items()
    }
    return unigram_model


# Function to estimate Bigram language model using MLE with Linear Interpolation Smoothing
def estimate_bigram(tokens):
    bigram_counts = Counter(ngrams(tokens, 2))
    unigram_counts = Counter(tokens)
    bigram_model = {}
    lambda_value = 0.9  # Smoothing parameter for linear interpolation
    for bigram, count in bigram_counts.items():
        prev_token = bigram[0]
        bigram_model[bigram] = (count + lambda_value) / (
            unigram_counts[prev_token] + lambda_value * len(unigram_counts)
        )
    return bigram_model


# Function to find top k words most likely to follow a given word in the Bigram language model
def find_top_words(bigram_model, word, k):
    candidates = [
        candidate for candidate in bigram_model.keys() if candidate[0] == word
    ]
    top_words = sorted(
        candidates, key=lambda x: bigram_model[x], reverse=True
    )[:k]
    return [word[1] for word in top_words]


# Read dataset
from my_io import my_io
from preproccessing import preproccessing

dataset_path = "./Dataset/"
dataset = my_io(dataset_path).read_jsons_from_folder()

# Preproccessing
tokens = preproccessing().tokenize(dataset)

print("H")


# Assuming you have a list of preprocessed tokens for each review in the variable 'preprocessed_reviews'
# Concatenate all tokens into a single list
all_tokens = [token for review_tokens in tokens for token in review_tokens]

# Estimate Unigram language model
unigram_model = estimate_unigram(all_tokens)

# Estimate Bigram language model with Linear Interpolation Smoothing
bigram_model = estimate_bigram(all_tokens)

# Find top 10 words most likely to follow the word 'decent' based on the Bigram language model
top_words_decent = find_top_words(bigram_model, "decent", 10)
