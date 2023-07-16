from collections import Counter
from nltk.util import ngrams
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer


class LM:
    def __init__(self, dataset):
        self.dataset = dataset

    # Function to estimate Unigram language model using MLE
    def estimate_unigram(self):
        flattened_tokens = list(chain.from_iterable(self.dataset))
        unigram_counts = Counter(flattened_tokens)
        total_tokens = len(flattened_tokens)
        unigram_model = {
            token: count / total_tokens
            for token, count in unigram_counts.items()
        }
        return unigram_model

    # Function to estimate Bigram language model using MLE with Linear
    # Interpolation Smoothing
    def estimate_bigram(self):
        flattened_tokens = list(chain.from_iterable(self.dataset))
        bigram_counts = Counter(ngrams(flattened_tokens, 2))
        unigram_counts = Counter(flattened_tokens)
        bigram_model = {}
        lambda_value = 0.9
        for bigram, count in bigram_counts.items():
            prev_token = bigram[0]
            bigram_model[bigram] = (count + lambda_value) / (
                unigram_counts[prev_token] + lambda_value * len(unigram_counts)
            )
        return bigram_model

    # Function to find top k words most likely to follow a given word in
    # the Bigram language model with Linear Interpolation Smoothing
    def find_top_words(
        self,
        bigram_model,
        unigram_model,
        tfidf_vectorizer,
        corpus,
        word,
        k,
        lambda_value,
    ):
        candidates = [
            candidate
            for candidate in bigram_model.keys()
            if candidate[0] == word
        ]

        # Calculate scores based on the probabilities from the models and TF-IDF values
        scores = []
        for candidate in candidates:
            prob_bigram = bigram_model[candidate].get(candidate[1], 0)
            prob_unigram = unigram_model.get(candidate[1], 0)
            tfidf_score = TfidfVectorizer.transform([corpus]).toarray()[0][
                tfidf_vectorizer.vocabulary_.get(candidate[1], 0)
            ]
            score = (
                lambda_value * prob_bigram
                + (1 - lambda_value) * prob_unigram
                + tfidf_score
            )
            scores.append((candidate[1], score))

        # Sort words based on scores in descending order
        sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

        # Return the top k words
        top_words = sorted_words[:k]

        return top_words
