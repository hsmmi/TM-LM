from collections import Counter
from nltk.util import ngrams
from itertools import chain


class LM:
    def __init__(self, dataset):
        self.dataset = dataset

    # Function to estimate Unigram language model using MLE
    def estimate_unigram(self):
        # Flatten the dataset and convert tokenized reviews to a list of words
        flattened_reviews = [
            word
            for resturant_review in self.dataset
            for review in resturant_review
            for word in review
        ]

        # Count the occurrences of each word in the dataset
        unigram_counts = {}
        for word in flattened_reviews:
            if word in unigram_counts:
                unigram_counts[word] += 1
            else:
                unigram_counts[word] = 1
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

        # Estimate the bigram model probabilities using MLE with
        # linear interpolation smoothing
        total_bigrams = sum(bigram_counts.values())
        bigram_model = {}
        for bigram, count in bigram_counts.items():
            bigram_prob = (count + 1) / (total_bigrams + len(bigram_counts))
            bigram_model[bigram] = bigram_prob

        return bigram_model

    def find_top_words(
        self, unigram_model, bigram_model, target_word, lambda_value, k
    ):
        candidates = [
            word2
            for word1, word2 in bigram_model.keys()
            if word1 == target_word
        ]

        normalized_bigram_model = [
            (word2, bigram_model[(target_word, word2)]) for word2 in candidates
        ]

        sum_bigram_model = sum([prob for _, prob in normalized_bigram_model])

        normalized_bigram_model = [
            (word2, prob / sum_bigram_model)
            for word2, prob in normalized_bigram_model
        ]
        normalized_bigram_model = dict(normalized_bigram_model)

        ###
        normalized_unigram_model = [
            (word2, unigram_model[word2]) for word2 in candidates
        ]

        sum_unigram_model = sum([prob for _, prob in normalized_unigram_model])

        normalized_unigram_model = [
            (word2, prob / sum_unigram_model)
            for word2, prob in normalized_unigram_model
        ]
        normalized_unigram_model = dict(normalized_unigram_model)

        ###
        scores = [
            (
                word2,
                lambda_value * normalized_unigram_model[word2]
                + (1 - lambda_value) * normalized_bigram_model[word2],
            )
            for word2 in candidates
        ]

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        top_scores = [scores[i][1] for i in top_indices]

        # Calculate the normalization factor
        normalization_factor = 100 / sum(top_scores)

        # Normalize the top scores
        normalized_scores = [
            score * normalization_factor for score in top_scores
        ]

        top_words_with_scores = [
            (candidates[candidate_indence], normalized_scores[score_indence])
            for score_indence, candidate_indence in enumerate(top_indices)
        ]

        top_words_with_scores = sorted(
            top_words_with_scores, key=lambda x: x[1], reverse=True
        )

        # Round the scores to 2 decimal places
        top_words_with_scores = [
            (word, round(score, 2)) for word, score in top_words_with_scores
        ]

        return top_words_with_scores
