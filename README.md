# Understanding the Co-occurrence Matrix Algorithm

At a high level, the algorithm's goal is to answer one question: **"For every word in my vocabulary, how many times does every other word appear within a certain distance (window size) of it?"**

Here is the step-by-step breakdown of the implementation:

## 1. The Setup: Vocabulary and Mapping
```python
words, num_words = distinct_words(corpus)
word2Ind = {word: index for index, word in enumerate(words)}
```
* **The Goal:** Matrices only understand numbers (row 0, column 1, etc.), not strings. We need a way to translate a word into a specific row/column index.
* **How it works:** You first get the sorted list of unique `words`. Then, you use `enumerate(words)` to pair each word with a number (e.g., `0: 'All'`, `1: "All's"`, etc.). The dictionary comprehension creates a fast lookup table.
* **Mental Note:** Think of `word2Ind` as the "directory" or "legend" for your matrix.

## 2. Initializing the Matrix
```python
M = np.zeros((num_words, num_words))
```
* **The Goal:** Create a blank slate to tally our counts.
* **How it works:** If you have V unique words, your matrix needs to be of size V x V. You use `np.zeros()` to create this square matrix filled with `0`s.
* **Mental Note:** `M[row, col]` will eventually hold the number of times the word corresponding to `row` appears next to the word corresponding to `col`.

## 3. Loop 1: Iterating Over Sentences
```python
for sentence in corpus:
    sentence_length = len(sentence)
```
* **The Goal:** Co-occurrence does not cross sentence boundaries. We must process the text sentence-by-sentence.
* **Mental Note:** We grab `sentence_length` here so we don't have to keep recalculating it later when figuring out our window boundaries.

## 4. Loop 2: Finding the Center Word
```python
    for center_i, center_word in enumerate(sentence):
        center_idx = word2Ind[center_word]
```
* **The Goal:** Look at every single word in the sentence, one by one, and treat it as the "center" of our universe for a brief moment.
* **How it works:** `enumerate()` gives us both the word and its position. We immediately look up `center_idx`, which tells us exactly which **row** in our matrix `M` we are going to be updating.

## 5. Calculating the Window Boundaries (The Tricky Part!)
```python
        start_i = max(0, center_i - window_size)
        end_i = min(sentence_length, center_i + window_size + 1)
```
* **The Goal:** Figure out where to start and stop looking for "context" words around our center word.
* **How it works:**
    * **Left Boundary (`start_i`):** Look `window_size` steps left. Using `max(0, ...)` acts as a floor, preventing negative indexing which would wrap around the list.
    * **Right Boundary (`end_i`):** Look `window_size` steps right. `+ 1` because Python's `range()` is exclusive at the end. `min(sentence_length, ...)` acts as a ceiling.

## 6. Loop 3: Iterating Over the Context Window
```python
        for context_i in range(start_i, end_i):
            if context_i == center_i:
                continue
```
* **The Goal:** Look at every word inside the calculated window.
* **The `if` statement:** Skip the center word itself so we don't count a word as co-occurring with itself, which inflates the diagonal of our matrix artificially.

## 7. Tallying the Co-occurrence
```python
            context_word = sentence[context_i]
            context_idx = word2Ind[context_word]
            M[center_idx, context_idx] += 1
```
* **The Goal:** We found a valid neighbor! Record it.
* **How it works:** Grab the string of the neighboring word, look up its matrix column index (`context_idx`), and add 1 to the tally at `M[row, column]`.

---

## Summary to Memorize
1. **Setup:** Make the dictionary mapping. Make the zeros matrix.
2. **Loop Sentences:** Go document by document.
3. **Loop Center Words:** Pick a word to be the focus. Look up its Matrix Row.
4. **Determine Boundaries:** `max(0, ...)` for the left, `min(length, ...)` for the right.
5. **Loop Context Words:** Look at neighbors. Skip yourself. Look up their Matrix Column. Add 1 to `M[row, col]`.
