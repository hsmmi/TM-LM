# Tokenization using NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download("punkt")


class preproccessing:
    def tokenize(self, dataset):
        tokenized_dataset = [
            [word_tokenize(review) for review in resturant_reviews]
            for resturant_reviews in dataset
        ]

        return tokenized_dataset

    def normalization(self, dataset):
        normalized_dataset = [
            [
                [token.lower() for token in review]
                for review in resturant_reviews
            ]
            for resturant_reviews in dataset
        ]
        return normalized_dataset

    def stemming(self, dataset):
        # Initialize Python porter stemmer
        ps = PorterStemmer()

        stemmed_dataset = [
            [
                [ps.stem(token) for token in review]
                for review in resturant_reviews
            ]
            for resturant_reviews in dataset
        ]

        return stemmed_dataset

    def remove_stop_word(self, dataset):
        # Initialize Python set of stopwords
        stop_words = set(nltk.corpus.stopwords.words("english"))

        cleaned_dataset = [
            [
                [token for token in review if token not in stop_words]
                for review in resturant_reviews
            ]
            for resturant_reviews in dataset
        ]

        # Remove empty strings
        cleaned_dataset = [
            [
                [token for token in review if token != ""]
                for review in resturant_reviews
            ]
            for resturant_reviews in cleaned_dataset
        ]

        return cleaned_dataset
