## The Theory Behind Truncated SVD

Truncated Singular Value Decomposition (SVD) is a linear dimensionality reduction technique. Similar to Principal Component Analysis (PCA), its goal is to reduce the number of features in a dataset while retaining as much of the original variance (information) as possible. 

However, Truncated SVD has a distinct advantage: **it does not require the data to be centered before processing.**

Because it operates directly on the raw data matrix, it is incredibly efficient for sparse matrices (matrices consisting mostly of zeros). With a solid foundation in machine learning and a move toward natural language processing, you will find Truncated SVD particularly useful for techniques like Latent Semantic Analysis (LSA). In LSA, text documents are represented as massive, sparse TF-IDF or count matrices, and Truncated SVD is used to distill them down to their core semantic topics without destroying the sparse data structure.


## The Mathematics of Truncated SVD

To understand Truncated SVD, we first need to look at standard SVD. SVD states that any rectangular $m \times n$ matrix $X$ can be factored into three distinct matrices:

$$X = U \Sigma V^T$$

Where:
* **$U$** is an $m \times m$ orthogonal matrix. Its columns are the left singular vectors.
* **$\Sigma$** is an $m \times n$ diagonal matrix containing the singular values of $X$ in descending order. These values represent the magnitude or "importance" of each corresponding vector.
* **$V^T$** is an $n \times n$ orthogonal matrix. Its rows are the right singular vectors.


### The "Truncation" Step

In a dataset with many features, many of the lower singular values in $\Sigma$ are close to zero, meaning they contribute very little to the overall structure of the data. **Truncated SVD** takes advantage of this by keeping only the top $k$ singular values, where $k$ is a user-defined hyperparameter strictly less than $\min(m, n)$.

By keeping only the top $k$ components, we approximate the original matrix $X$:

$$X \approx X_k = U_k \Sigma_k V_k^T$$

Where:
* **$U_k$** is now $m \times k$.
* **$\Sigma_k$** is a $k \times k$ diagonal matrix.
* **$V_k^T$** is $k \times n$.

To project our original data into this new, lower-dimensional space, we compute the dot product of the original matrix and the truncated right singular vectors:

$$X_{reduced} = X V_k$$

Alternatively, using the SVD components, the transformed data is simply $U_k \Sigma_k$. 

---

## Using `sklearn.decomposition.TruncatedSVD`

Scikit-learn makes this highly accessible. Because Truncated SVD is so prominent in text processing, the following Python script demonstrates how to use it on a sparse matrix generated from text data.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1. Prepare some sample text data
documents = [
    "Machine learning relies heavily on linear algebra.",
    "Natural language processing bridges human communication and data.",
    "Linear algebra and calculus are fundamental to algorithms.",
    "Deep learning models require significant computational power."
]

# 2. Convert text to a sparse TF-IDF matrix
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(documents)

print(f"Original sparse matrix shape: {X_sparse.shape}")

# 3. Initialize TruncatedSVD
# n_components is our 'k' from the mathematical equation
k = 2 
svd = TruncatedSVD(n_components=k, random_state=42)

# 4. Fit the model and transform the data
X_reduced = svd.fit_transform(X_sparse)

print(f"Reduced dense matrix shape: {X_reduced.shape}\n")

# 5. Analyze the results
print("Explained Variance Ratio per component:")
print(np.round(svd.explained_variance_ratio_, 3))

print("\nTotal Explained Variance:")
print(f"{np.sum(svd.explained_variance_ratio_) * 100:.2f}%")
```

### Key Parameters in Scikit-Learn
* **`n_components`**: The target dimensionality ($k$). Choosing this usually involves a trade-off between computational efficiency and information retention.
* **`algorithm`**: Defaults to `'randomized'`, which uses a fast randomized SVD solver (perfect for massive datasets). You can also set it to `'arpack'` for a more exact, albeit slower, calculation.

---

## Computing $X_{reduced}$ by Hand

To compute your reduced matrix $X_{reduced}$, you must first find the right singular vectors that make up $V_k$. In standard mathematics, finding the components of an SVD involves calculating the eigenvectors and eigenvalues of a matrix. Because we are looking specifically for $V$ (the right singular vectors), we need to look at the relationship between your original matrix $X$ and its transpose $X^T$.

### Step 1: Compute the Product $X^T X$

To find the right singular vectors, we must multiply the transpose of your original matrix by the original matrix itself. If $X$ is an $m \times n$ matrix (where $m$ is the number of documents/samples and $n$ is the number of features), then $X^T$ is $n \times m$. Multiplying them together yields a square $n \times n$ matrix:

$$M = X^T X$$

This resulting square matrix $M$ captures the correlations between the different features in your dataset.

### Step 2: Find the Eigenvalues and Eigenvectors

Next, you must find the eigenvalues ($\lambda$) and eigenvectors ($v$) of your new square matrix $M$. This is the most computationally heavy part of doing it by hand. You find the eigenvalues by solving the characteristic equation:

$$\det(M - \lambda I) = 0$$

Where $I$ is the identity matrix. Once you have your eigenvalues, you plug each one back into the following equation to solve for its corresponding eigenvector $v$:

$$(M - \lambda I)v = 0$$


### Step 3: Sort and Form Matrix $V$

The eigenvectors you just calculated are the right singular vectors of $X$. However, to truncate them properly, order matters.

* **Calculate Singular Values:** The singular values of $X$ (which make up the diagonal matrix $\Sigma$) are simply the square roots of your eigenvalues: $\sigma = \sqrt{\lambda}$.
* **Sort:** Sort your eigenvalues (and their corresponding singular values) in descending order, from largest to smallest.
* **Construct $V$:** Arrange your eigenvectors as columns in a new matrix $V$, matching the descending order of their corresponding eigenvalues. The first column corresponds to the largest eigenvalue, the second column to the second largest, and so on.

### Step 4: Truncate to $V_k$

Now comes the "Truncated" part of Truncated SVD. Since your eigenvectors are sorted by how much variance (information) they capture, the columns on the right side of matrix $V$ represent noise or highly minor details. 

Choose your $k$ value (the target number of dimensions). Keep the first $k$ columns of $V$ and discard the rest. This leaves you with an $n \times k$ matrix:

$$V_k$$

### Step 5: Project into the Reduced Space

Finally, you project your original sparse data $X$ into this new, dense, lower-dimensional space by taking the dot product of $X$ and $V_k$:

$$X_{reduced} = X V_k$$

Because $X$ is $m \times n$ and $V_k$ is $n \times k$, your resulting $X_{reduced}$ matrix will be $m \times k$. You have successfully reduced the features from $n$ to $k$ while preserving the most important relationships in the data!

---

## Walkthrough: Eigenvalues and Eigenvectors of a 2x2 Matrix

Let’s walk through a concrete example. Because $M = X^T X$ will always result in a square, symmetric matrix, let's use a simple 2x2 symmetric matrix for $M$:

$$M = \begin{pmatrix} 4 & 2 \\ 2 & 4 \end{pmatrix}$$

Here is how we extract the eigenvalues and their corresponding eigenvectors step-by-step.

### Step 1: Solving the Characteristic Equation (Finding $\lambda$)

Our goal is to find the scalar values ($\lambda$) where subtracting them from the diagonal of our matrix $M$ makes the determinant equal to zero. This is the characteristic equation:

$$\det(M - \lambda I) = 0$$

First, we set up our matrix inside the determinant by subtracting $\lambda$ from the diagonal elements:

$$\det \begin{pmatrix} 4 - \lambda & 2 \\ 2 & 4 - \lambda \end{pmatrix} = 0$$

To find the determinant of a 2x2 matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, we calculate $(ad - bc)$. Applying that here gives us:

$$(4 - \lambda)(4 - \lambda) - (2)(2) = 0$$

Now, we expand and simplify the algebra:

$$16 - 4\lambda - 4\lambda + \lambda^2 - 4 = 0$$
$$\lambda^2 - 8\lambda + 12 = 0$$

This leaves us with a standard quadratic equation. We can factor this by finding two numbers that multiply to $12$ and add to $-8$:

$$(\lambda - 6)(\lambda - 2) = 0$$

Solving for $\lambda$ gives us our two eigenvalues:
* **$\lambda_1 = 6$**
* **$\lambda_2 = 2$**


### Step 2: Solving for the Eigenvectors ($v$)

Now that we have our eigenvalues, we plug them one at a time back into the equation $(M - \lambda I)v = 0$ to find the corresponding eigenvector $v = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}$.

#### Finding the first eigenvector (using $\lambda_1 = 6$)

Substitute $6$ for $\lambda$:

$$\begin{pmatrix} 4 - 6 & 2 \\ 2 & 4 - 6 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$$\begin{pmatrix} -2 & 2 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives us a system of linear equations:
1.  $-2v_1 + 2v_2 = 0$
2.  $2v_1 - 2v_2 = 0$

Both equations simplify to the exact same relationship: $v_1 = v_2$. This means any vector where the first and second components are equal is an eigenvector for $\lambda=6$. The simplest whole-number representation is:

$$v_{\lambda1} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

#### Finding the second eigenvector (using $\lambda_2 = 2$)

Substitute $2$ for $\lambda$:

$$\begin{pmatrix} 4 - 2 & 2 \\ 2 & 4 - 2 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$$\begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives us the system:
1.  $2v_1 + 2v_2 = 0$
2.  $2v_1 + 2v_2 = 0$

This simplifies to $v_1 = -v_2$. The simplest whole-number representation here is:

$$v_{\lambda2} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

### Step 3: Normalizing the Vectors (Crucial for SVD)

In Singular Value Decomposition, the matrix $V$ must be orthogonal, meaning its column vectors need to have a length (magnitude) of exactly 1. 

To normalize a vector, we divide each component by the vector's total length. The length is found using the Pythagorean theorem ($\sqrt{v_1^2 + v_2^2}$).

For our first vector $\begin{pmatrix} 1 \\ 1 \end{pmatrix}$, the length is $\sqrt{1^2 + 1^2} = \sqrt{2}$. 
* **Normalized $v_1$:** $\begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix}$

For our second vector $\begin{pmatrix} 1 \\ -1 \end{pmatrix}$, the length is $\sqrt{1^2 + (-1)^2} = \sqrt{2}$.
* **Normalized $v_2$:** $\begin{pmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{pmatrix}$

These normalized vectors are the right singular vectors that will become the columns of your $V$ matrix!

### Step 4: Constructing Matrix $V$

In our previous steps, we found our two eigenvalues and their corresponding normalized eigenvectors:
* For $\lambda_1 = 6$, our vector is $v_1 = \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix}$
* For $\lambda_2 = 2$, our vector is $v_2 = \begin{pmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{pmatrix}$

To build the matrix $V$, we place these vectors as columns. **Crucially, they must be ordered from the largest eigenvalue to the smallest.** Since $6 > 2$, $v_1$ becomes our first column and $v_2$ becomes our second.

$$V = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}$$


### Step 5: Applying the Truncation ($V_k$)

Our original dataset had 2 features (since $M$ was $2 \times 2$). Let's say we want to reduce our feature space from 2 dimensions down to 1 dimension. This means we set our hyperparameter $k = 1$.

To truncate $V$ into $V_k$, we simply keep the first $k$ columns and discard the rest. By keeping only the first column, we retain the vector associated with the largest eigenvalue (the one that captures the most variance in the data).

$$V_1 = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}$$

We have successfully truncated our right singular matrix!

### Step 6: Projecting the Data ($X_{reduced}$)

Now we project a hypothetical dataset into this new 1-dimensional space. Let's assume our original data matrix $X$ consists of 3 samples (rows) and 2 features (columns):

$$X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$$

To find our reduced matrix, we compute the dot product of $X$ and our truncated matrix $V_1$:

$$X_{reduced} = X V_1$$

Let's do the matrix multiplication. We multiply the rows of $X$ by the single column of $V_1$:

$$X_{reduced} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix} \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix}$$

* **Row 1:** $(1 \cdot \frac{1}{\sqrt{2}}) + (2 \cdot \frac{1}{\sqrt{2}}) = \frac{3}{\sqrt{2}}$
* **Row 2:** $(3 \cdot \frac{1}{\sqrt{2}}) + (4 \cdot \frac{1}{\sqrt{2}}) = \frac{7}{\sqrt{2}}$
* **Row 3:** $(5 \cdot \frac{1}{\sqrt{2}}) + (6 \cdot \frac{1}{\sqrt{2}}) = \frac{11}{\sqrt{2}}$


Putting it all together, our final reduced data matrix is:

$$X_{reduced} = \begin{pmatrix} \frac{3}{\sqrt{2}} \\ \frac{7}{\sqrt{2}} \\ \frac{11}{\sqrt{2}} \end{pmatrix}$$

#### The Result
We started with a dataset that had 3 samples and **2 features** (a $3 \times 2$ matrix). After finding the eigenvalues, constructing $V$, truncating to $k=1$, and computing the dot product, we now have a dataset with 3 samples and only **1 feature** (a $3 \times 1$ matrix). 

You have successfully performed dimensionality reduction by hand using the mathematical foundations of SVD!
