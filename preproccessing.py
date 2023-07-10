# Tokenization using NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# deep copy
import copy

nltk.download("punkt")


class preproccessing:
    def tokenize(self, dataset):
        # deep copy dataset
        dataset = copy.deepcopy(dataset)
        for n_resturant in range(len(dataset)):
            for n_review in range(len(dataset[n_resturant])):
                dataset[n_resturant][n_review] = word_tokenize(
                    dataset[n_resturant][n_review]
                )

        return dataset

    def normalization(self, dataset):
        # deep copy dataset
        dataset = copy.deepcopy(dataset)
        for n_resturant in range(len(dataset)):
            for n_review in range(len(dataset[n_resturant])):
                for n_token in range(len(dataset[n_resturant][n_review])):
                    dataset[n_resturant][n_review][n_token] = dataset[
                        n_resturant
                    ][n_review][n_token].lower()

        return dataset

    def stemming(self, dataset):
        # deep copy dataset
        dataset = copy.deepcopy(dataset)

        # Initialize Python porter stemmer
        ps = PorterStemmer()
        for n_resturant in range(len(dataset)):
            for n_review in range(len(dataset[n_resturant])):
                for n_token in range(len(dataset[n_resturant][n_review])):
                    dataset[n_resturant][n_review][n_token] = ps.stem(
                        dataset[n_resturant][n_review][n_token]
                    )

    def remove_stop_word(self, dataset):
        # deep copy dataset
        dataset = copy.deepcopy(dataset)

        stop_words = set(nltk.corpus.stopwords.words("english"))
        for n_resturant in range(len(dataset)):
            for n_review in range(len(dataset[n_resturant])):
                for n_token in range(len(dataset[n_resturant][n_review])):
                    if dataset[n_resturant][n_review][n_token] in stop_words:
                        dataset[n_resturant][n_review][n_token] = ""

        # remove ""
        for resturant in dataset:
            for review in resturant:
                while "" in review:
                    review.remove("")

        return dataset
