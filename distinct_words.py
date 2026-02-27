def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = 0

    distinct_words = {word for sublist in corpus for word in sublist}
    distinct_words_list = list(distinct_words)
    corpus_words = sorted(distinct_words_list)
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words
