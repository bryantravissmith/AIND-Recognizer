import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
	""" Recognize test word sequences from word models set
	:param models: dict of trained model
		{'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
		:param test_set: SinglesData object
	:return: (list, list)  as probabilities, guesses
		both lists are ordered by the test set word_id
		probabilities is a list of dictionaries where each key a word and value is Log Liklihood
			[{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			 {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			 ]
		guesses is a list of the best guess words ordered by the test set word_id
			['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
	"""
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	warnings.filterwarnings("ignore", category=RuntimeWarning)

	probabilities = []
	guesses = []
	for index, (X, length) in test_set.get_all_Xlengths().items():
		probs = []
		for word, mod in models.items():
			try:
				probs.append((word, mod.score(X, length)))
			except:
				probs.append((word, float("-inf")))
		most_probable_word = sorted(probs, key=lambda x: -x[1])[0][0]

		probabilities.append(dict(probs))
		guesses.append(most_probable_word)

	return probabilities, guesses

def bigram_recognize(models: dict, test_set: SinglesData, unigram: dict, bigram: dict):
	""" Recognize test word sequences from word models set
	:param models: dict of trained model
		{'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
	:param test_set: SinglesData object
	:param unigram: Dict of the probability of each word a training corpus
	:param bigram: Dictionary of counts for each bigram 
	:return: (list, list)  as probabilities, guesses
		both lists are ordered by the test set word_id
		probabilities is a list of dictionaries where each key a word and value is Log Liklihood
			[{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			 {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			 ]
		guesses is a list of the best guess words ordered by the test set word_id
			['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
	"""
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	warnings.filterwarnings("ignore", category=RuntimeWarning)

	probabilities, guesses = recognize(models, test_set)

	new_guesses = guesses.copy()
	new_probabilities = []
	for sentence_number, indexes in test_set.sentences_index.items():
		for i in range(0, len(indexes)):
			if i == 0:
				new_prob = [(word, logl + np.log(unigram[word])) for word, logl in probabilities[indexes[i]].items()]
				new_probabilities.append(dict(new_prob))
				new_guess = sorted(new_prob, key=lambda x: -x[1])[0][0]
				new_guesses[indexes[i]] = new_guess
			else:
				previous_word = new_guesses[indexes[i - 1]]
				total = sum(bigram[previous_word].values())
				if total > 0:
					new_prob = [(word, logl + np.log(bigram[previous_word][word] / total)) for word, logl in probabilities[indexes[i]].items()]
					new_probabilities.append(dict(new_prob))
					new_guess = sorted(new_prob, key=lambda x: -x[1])[0][0]
					new_guesses[indexes[i]] = new_guess

	return new_probabilities, new_guesses
