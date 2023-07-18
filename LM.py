from collections import Counter
from nltk.util import ngrams
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer


class LM:
    def __init__(self, dataset):
        self.dataset = dataset

    # Function to estimate Unigram language model using MLE
    def estimate_unigram(self):
        # Flatten the dataset and convert tokenized reviews to a list of words
        flattened_reviews = [
            word for review in self.dataset for word in review
        ]

        # Count the occurrences of each word in the dataset
        unigram_counts = {}
        for word in flattened_reviews:
            unigram_counts[word] = unigram_counts.get(word, 0) + 1

        total_words = len(flattened_reviews)

        # Estimate the unigram model probabilities using MLE
        unigram_model = {
            word: count / total_words for word, count in unigram_counts.items()
        }

        return unigram_model

    # Function to estimate Bigram language model using MLE with Linear
    # Interpolation Smoothing
    from collections import Counter

    def estimate_bigram(self, lambda_value=0.9):
        # Flatten the dataset and convert tokenized reviews to strings
        flattened_reviews = [
            " ".join(review) for review in chain.from_iterable(self.dataset)
        ]

        # Create a dictionary to store bigram counts
        bigram_counts = Counter()

        # Count bigrams in the dataset
        for review in flattened_reviews:
            tokens = review.split()
            bigrams = ngrams(tokens, 2)
            bigram_counts.update(bigrams)

        # Estimate the bigram model probabilities using MLE with linear interpolation smoothing
        total_bigrams = sum(bigram_counts.values())
        bigram_model = {}
        for bigram, count in bigram_counts.items():
            bigram_prob = (count + 1) / (total_bigrams + len(bigram_counts))
            bigram_model[bigram] = bigram_prob

        return bigram_model

    # def estimate_bigram(self, lambda_value=0.9):

    #     # Flatten the dataset and convert tokenized reviews to strings
    #     flattened_reviews = [
    #         " ".join(review) for review in chain.from_iterable(self.dataset)
    #     ]

    #     # Create a dictionary to store bigram counts
    #     bigram_counts = Counter()

    #     # Iterate over each document and update the bigram counts
    #     for review in flattened_reviews:
    #         tokens = review.split()
    #         bigrams = ngrams(tokens, 2)
    #         bigram_counts.update(bigrams)

    #     # Estimate the bigram model probabilities using MLE with
    #     # linear interpolation smoothing
    #     bigram_model = {}
    #     for bigram, count in bigram_counts.items():
    #         word1, word2 = bigram
    #         unigram_count = sum(
    #             [count for _, count in bigram_counts.items() if _ == word1]
    #         )
    #         bigram_prob = (count + 1) / (unigram_count + len(bigram_counts))
    #         unigram_prob = (unigram_count + 1) / (
    #             sum(bigram_counts.values()) + len(bigram_counts)
    #         )
    #         interpolated_prob = (
    #             lambda_value * bigram_prob + (1 - lambda_value) * unigram_prob
    #         )
    #         bigram_model[bigram] = interpolated_prob

    #     return bigram_model

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

        # Calculate scores based on the probabilities from the models
        # and TF-IDF values
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
