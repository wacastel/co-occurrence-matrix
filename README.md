\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

% Setup custom colors for the code blocks
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

% Setup the style for the code blocks
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=4
}
\lstset{style=mystyle}

\title{\textbf{Understanding the Co-occurrence Matrix Algorithm}}
\author{Algorithm Breakdown Reference}
\date{}

\begin{document}

\maketitle

At a high level, the algorithm's goal is to answer one question: \textbf{"For every word in my vocabulary, how many times does every other word appear within a certain distance (window size) of it?"}

Here is the step-by-step breakdown of the implementation:

\section*{1. The Setup: Vocabulary and Mapping}
\begin{lstlisting}[language=Python]
words, num_words = distinct_words(corpus)
word2Ind = {word: index for index, word in enumerate(words)}
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Matrices only understand numbers (row 0, column 1, etc.), not strings. We need a way to translate a word into a specific row/column index.
    \item \textbf{How it works:} You first get the sorted list of unique \texttt{words}. Then, you use \texttt{enumerate(words)} to pair each word with a number (e.g., 0: 'All', 1: "All's", etc.). The dictionary comprehension creates a fast lookup table.
    \item \textbf{Mental Note:} Think of \texttt{word2Ind} as the "directory" or "legend" for your matrix.
\end{itemize}

\section*{2. Initializing the Matrix}
\begin{lstlisting}[language=Python]
M = np.zeros((num_words, num_words))
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Create a blank slate to tally our counts.
    \item \textbf{How it works:} If you have $V$ unique words, your matrix needs to be of size $V \times V$. You use \texttt{np.zeros()} to create this square matrix filled with 0s.
    \item \textbf{Mental Note:} \texttt{M[row, col]} will eventually hold the number of times the word corresponding to \texttt{row} appears next to the word corresponding to \texttt{col}.
\end{itemize}

\section*{3. Loop 1: Iterating Over Sentences}
\begin{lstlisting}[language=Python]
for sentence in corpus:
    sentence_length = len(sentence)
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Co-occurrence does not cross sentence boundaries. We must process the text sentence-by-sentence.
    \item \textbf{Mental Note:} We grab \texttt{sentence\_length} here so we don't have to keep recalculating it later when figuring out our window boundaries.
\end{itemize}

\section*{4. Loop 2: Finding the Center Word}
\begin{lstlisting}[language=Python]
    for center_i, center_word in enumerate(sentence):
        center_idx = word2Ind[center_word]
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Look at every single word in the sentence, one by one, and treat it as the "center" of our universe for a brief moment.
    \item \textbf{How it works:} \texttt{enumerate()} gives us both the word and its position. We immediately look up \texttt{center\_idx}, which tells us exactly which \textbf{row} in our matrix M we are going to be updating.
\end{itemize}

\section*{5. Calculating the Window Boundaries (The Tricky Part!)}
\begin{lstlisting}[language=Python]
        start_i = max(0, center_i - window_size)
        end_i = min(sentence_length, center_i + window_size + 1)
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Figure out where to start and stop looking for "context" words around our center word.
    \item \textbf{How it works:}
    \begin{itemize}
        \item \textbf{Left Boundary (\texttt{start\_i}):} Look \texttt{window\_size} steps left. Using \texttt{max(0, ...)} acts as a floor, preventing negative indexing which would wrap around the list.
        \item \textbf{Right Boundary (\texttt{end\_i}):} Look \texttt{window\_size} steps right. \texttt{+ 1} because Python's \texttt{range()} is exclusive at the end. \texttt{min(sentence\_length, ...)} acts as a ceiling.
    \end{itemize}
\end{itemize}

\section*{6. Loop 3: Iterating Over the Context Window}
\begin{lstlisting}[language=Python]
        for context_i in range(start_i, end_i):
            if context_i == center_i:
                continue
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} Look at every word inside the calculated window.
    \item \textbf{The \texttt{if} statement:} Skip the center word itself so we don't count a word as co-occurring with itself, which inflates the diagonal of our matrix artificially.
\end{itemize}

\section*{7. Tallying the Co-occurrence}
\begin{lstlisting}[language=Python]
            context_word = sentence[context_i]
            context_idx = word2Ind[context_word]
            M[center_idx, context_idx] += 1
\end{lstlisting}
\begin{itemize}
    \item \textbf{The Goal:} We found a valid neighbor! Record it.
    \item \textbf{How it works:} Grab the string of the neighboring word, look up its matrix column index (\texttt{context\_idx}), and add 1 to the tally at \texttt{M[row, column]}.
\end{itemize}

\section*{Summary to Memorize}
\begin{enumerate}
    \item \textbf{Setup:} Make the dictionary mapping. Make the zeros matrix.
    \item \textbf{Loop Sentences:} Go document by document.
    \item \textbf{Loop Center Words:} Pick a word to be the focus. Look up its Matrix Row.
    \item \textbf{Determine Boundaries:} \texttt{max(0, ...)} for the left, \texttt{min(length, ...)} for the right.
    \item \textbf{Loop Context Words:} Look at neighbors. Skip yourself. Look up their Matrix Column. Add 1 to \texttt{M[row, col]}.
\end{enumerate}

\end{document}
