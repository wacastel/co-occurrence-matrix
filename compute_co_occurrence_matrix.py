import numpy as np

def create_co_occurrence_matrix(corpus, window_size, sorted_distinct_word_list, num_words):
    """
    Computes a co-occurrence matrix for a given corpus and vocabulary.
    
    Args:
        corpus: List of lists of strings (e.g., [['the', 'cat', ...], ...])
        window_size: Integer, size of the context window (left and right)
        sorted_distinct_word_list: List of unique words sorted alphabetically
        num_words: Integer, number of unique words
        
    Returns:
        M: numpy matrix of shape (num_words, num_words)
        word2Ind: dictionary mapping word to its matrix index
    """
    
    # 1. Map words to indices based on the sorted list
    word2Ind = {word: index for index, word in enumerate(sorted_distinct_word_list)}
    
    # 2. Initialize the co-occurrence matrix M with zeros
    M = np.zeros((num_words, num_words))
    
    # 3. Iterate through the corpus to count co-occurrences
    for sentence in corpus:
        sentence_length = len(sentence)
        
        for center_i, center_word in enumerate(sentence):
            # Get the matrix index for the center word
            center_idx = word2Ind[center_word]
            
            # Define the bounds of the context window
            start_i = max(0, center_i - window_size)
            end_i = min(sentence_length, center_i + window_size + 1)
            
            # Iterate through the context words within the sliding window
            for context_i in range(start_i, end_i):
                if context_i == center_i:
                    continue # Skip the center word itself
                
                context_word = sentence[context_i]
                context_idx = word2Ind[context_word]
                
                # Increment the co-occurrence count
                M[center_idx, context_idx] += 1
                    
    return M, word2Ind
