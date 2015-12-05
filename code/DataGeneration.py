# This is generating Text data from corpus
# http://www.nltk.org/book/ch02.html for detailed 
import nltk
from nltk.corpus import brown
import numpy as np

# The cateogires include
# news, editorial, reviews, religion, hobbies, lore, belles_lettres
# government, learned, fiction, mystery, science_fiction, adventure, romance, humor
def brown_corpus(category):
	""" Imports Brown text data for the given category"""
	return nltk.pos_tag([str(word) for word in brown.words(categories = category)])

def brown_corpus_input():
	""" Imports Brown text data by selecting categories from the commandline"""
	print "Brown corpus include following categories"
	print [str(category) for category in brown.categories()]
	category_set = set([str(category) for category in brown.categories()])

	response = raw_input("\n Please enter the category to import \n Category: ")

	while response not in category_set:
		response = raw_input("\n Please re-enter the category to import \n Category: ")

	return nltk.pos_tag([str(word) for word in brown.words(categories = response)])


def main():
	return brown_corpus_input()

if __name__ == '__main__':
    corpus = main()
    vocab = list(set([r[0] for r in corpus]))
    categories = list(set([r[1] for r in corpus]))
    corpus = [np.array([vocab.index(r[0]), categories.index(r[1])]) for r in corpus]
    test_ln = 1000
    train_ln = len(corpus) - test_ln
    train_corpus = corpus[:train_ln]
    test_corpus = corpus[train_ln:]
    train_corpus0 = train_corpus[:-1]
    train_corpus1 = train_corpus[1:]
    test_corpus0 = test_corpus[:-1]
    test_corpus1 = test_corpus[1:]
