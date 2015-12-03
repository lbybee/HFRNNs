# This is generating Text data from corpus
# http://www.nltk.org/book/ch02.html for detailed 
import nltk
from nltk.corpus import brown

# The cateogires include
# news, editorial, reviews, religion, hobbies, lore, belles_lettres
# government, learned, fiction, mystery, science_fiction, adventure, romance, humor
def brown_corpus(category):
	return nltk.pos_tag([str(word) for word in brown.words(categories = category)])

def brown_corpus_input():
	print "Brown corpus include following categories"
	print [str(category) for category in brown.categories()]
	category_set = set([str(category) for category in brown.categories()])

	response = raw_input("\n Please enter the category to import \n Category: ")

	while response not in category_set:
		response = raw_input("\n Please re-enter the category to import \n Category: ")

	return nltk.pos_tag([str(word) for word in brown.words(categories = response)])


def main():
	print brown_corpus_input()

if __name__ == '__main__':
    main()
