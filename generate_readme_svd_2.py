import os

def generate_readme():
    # Variables used to preserve markdown code blocks and LaTeX delimiters
    # when writing the string to the file.
    py_code_block = "```python"
    end_code_block = "```"
    d = "$"
    dd = "$$"

    readme_content = rf"""## The Theory Behind Truncated SVD

Truncated Singular Value Decomposition (SVD) is a linear dimensionality reduction technique. Similar to Principal Component Analysis (PCA), its goal is to reduce the number of features in a dataset while retaining as much of the original variance (information) as possible. 

However, Truncated SVD has a distinct advantage: **it does not require the data to be centered before processing.**

Because it operates directly on the raw data matrix, it is incredibly efficient for sparse matrices (matrices consisting mostly of zeros). With a solid foundation in machine learning and a move toward natural language processing, you will find Truncated SVD particularly useful for techniques like Latent Semantic Analysis (LSA). In LSA, text documents are represented as massive, sparse TF-IDF or count matrices, and Truncated SVD is used to distill them down to their core semantic topics without destroying the sparse data structure.


## The Mathematics of Truncated SVD

To understand Truncated SVD, we first need to look at standard SVD. SVD states that any rectangular {d}m \times n{d} matrix {d}X{d} can be factored into three distinct matrices:

{dd}X = U \Sigma V^T{dd}

Where:
* **{d}U{d}** is an {d}m \times m{d} orthogonal matrix. Its columns are the left singular vectors.
* **{d}\Sigma{d}** is an {d}m \times n{d} diagonal matrix containing the singular values of {d}X{d} in descending order. These values represent the magnitude or "importance" of each corresponding vector.
* **{d}V^T{d}** is an {d}n \times n{d} orthogonal matrix. Its rows are the right singular vectors.


### The "Truncation" Step

In a dataset with many features, many of the lower singular values in {d}\Sigma{d} are close to zero, meaning they contribute very little to the overall structure of the data. **Truncated SVD** takes advantage of this by keeping only the top {d}k{d} singular values, where {d}k{d} is a user-defined hyperparameter strictly less than {d}\min(m, n){d}.

By keeping only the top {d}k{d} components, we approximate the original matrix {d}X{d}:

{dd}X \approx X_k = U_k \Sigma_k V_k^T{dd}

Where:
* **{d}U_k{d}** is now {d}m \times k{d}.
* **{d}\Sigma_k{d}** is a {d}k \times k{d} diagonal matrix.
* **{d}V_k^T{d}** is {d}k \times n{d}.

To project our original data into this new, lower-dimensional space, we compute the dot product of the original matrix and the truncated right singular vectors:

{dd}X_{{reduced}} = X V_k{dd}

Alternatively, using the SVD components, the transformed data is simply {d}U_k \Sigma_k{d}. 

---

## Using `sklearn.decomposition.TruncatedSVD`

Scikit-learn makes this highly accessible. Because Truncated SVD is so prominent in text processing, the following Python script demonstrates how to use it on a sparse matrix generated from text data.

{py_code_block}
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

print(f"Original sparse matrix shape: {{X_sparse.shape}}")

# 3. Initialize TruncatedSVD
# n_components is our 'k' from the mathematical equation
k = 2 
svd = TruncatedSVD(n_components=k, random_state=42)

# 4. Fit the model and transform the data
X_reduced = svd.fit_transform(X_sparse)

print(f"Reduced dense matrix shape: {{X_reduced.shape}}\n")

# 5. Analyze the results
print("Explained Variance Ratio per component:")
print(np.round(svd.explained_variance_ratio_, 3))

print("\nTotal Explained Variance:")
print(f"{{np.sum(svd.explained_variance_ratio_) * 100:.2f}}%")
{end_code_block}

### Key Parameters in Scikit-Learn
* **`n_components`**: The target dimensionality ({d}k{d}). Choosing this usually involves a trade-off between computational efficiency and information retention.
* **`algorithm`**: Defaults to `'randomized'`, which uses a fast randomized SVD solver (perfect for massive datasets). You can also set it to `'arpack'` for a more exact, albeit slower, calculation.

---

## Computing {d}X_{{reduced}}{d} by Hand

To compute your reduced matrix {d}X_{{reduced}}{d}, you must first find the right singular vectors that make up {d}V_k{d}. In standard mathematics, finding the components of an SVD involves calculating the eigenvectors and eigenvalues of a matrix. Because we are looking specifically for {d}V{d} (the right singular vectors), we need to look at the relationship between your original matrix {d}X{d} and its transpose {d}X^T{d}.

### Step 1: Compute the Product {d}X^T X{d}

To find the right singular vectors, we must multiply the transpose of your original matrix by the original matrix itself. If {d}X{d} is an {d}m \times n{d} matrix (where {d}m{d} is the number of documents/samples and {d}n{d} is the number of features), then {d}X^T{d} is {d}n \times m{d}. Multiplying them together yields a square {d}n \times n{d} matrix:

{dd}M = X^T X{dd}

This resulting square matrix {d}M{d} captures the correlations between the different features in your dataset.

### Step 2: Find the Eigenvalues and Eigenvectors

Next, you must find the eigenvalues ({d}\lambda{d}) and eigenvectors ({d}v{d}) of your new square matrix {d}M{d}. This is the most computationally heavy part of doing it by hand. You find the eigenvalues by solving the characteristic equation:

{dd}\det(M - \lambda I) = 0{dd}

Where {d}I{d} is the identity matrix. Once you have your eigenvalues, you plug each one back into the following equation to solve for its corresponding eigenvector {d}v{d}:

{dd}(M - \lambda I)v = 0{dd}


### Step 3: Sort and Form Matrix {d}V{d}

The eigenvectors you just calculated are the right singular vectors of {d}X{d}. However, to truncate them properly, order matters.

* **Calculate Singular Values:** The singular values of {d}X{d} (which make up the diagonal matrix {d}\Sigma{d}) are simply the square roots of your eigenvalues: {d}\sigma = \sqrt{{\lambda}}{d}.
* **Sort:** Sort your eigenvalues (and their corresponding singular values) in descending order, from largest to smallest.
* **Construct {d}V{d}:** Arrange your eigenvectors as columns in a new matrix {d}V{d}, matching the descending order of their corresponding eigenvalues. The first column corresponds to the largest eigenvalue, the second column to the second largest, and so on.

### Step 4: Truncate to {d}V_k{d}

Now comes the "Truncated" part of Truncated SVD. Since your eigenvectors are sorted by how much variance (information) they capture, the columns on the right side of matrix {d}V{d} represent noise or highly minor details. 

Choose your {d}k{d} value (the target number of dimensions). Keep the first {d}k{d} columns of {d}V{d} and discard the rest. This leaves you with an {d}n \times k{d} matrix:

{dd}V_k{dd}

### Step 5: Project into the Reduced Space

Finally, you project your original sparse data {d}X{d} into this new, dense, lower-dimensional space by taking the dot product of {d}X{d} and {d}V_k{d}:

{dd}X_{{reduced}} = X V_k{dd}

Because {d}X{d} is {d}m \times n{d} and {d}V_k{d} is {d}n \times k{d}, your resulting {d}X_{{reduced}}{d} matrix will be {d}m \times k{d}. You have successfully reduced the features from {d}n{d} to {d}k{d} while preserving the most important relationships in the data!

---

## Walkthrough: Eigenvalues and Eigenvectors of a 2x2 Matrix

Let’s walk through a concrete example. Because {d}M = X^T X{d} will always result in a square, symmetric matrix, let's use a simple 2x2 symmetric matrix for {d}M{d}:

{dd}M = \begin{{pmatrix}} 4 & 2 \\ 2 & 4 \end{{pmatrix}}{dd}

Here is how we extract the eigenvalues and their corresponding eigenvectors step-by-step.

### Step 1: Solving the Characteristic Equation (Finding {d}\lambda{d})

Our goal is to find the scalar values ({d}\lambda{d}) where subtracting them from the diagonal of our matrix {d}M{d} makes the determinant equal to zero. This is the characteristic equation:

{dd}\det(M - \lambda I) = 0{dd}

First, we set up our matrix inside the determinant by subtracting {d}\lambda{d} from the diagonal elements:

{dd}\det \begin{{pmatrix}} 4 - \lambda & 2 \\ 2 & 4 - \lambda \end{{pmatrix}} = 0{dd}

To find the determinant of a 2x2 matrix {d}\begin{{pmatrix}} a & b \\ c & d \end{{pmatrix}}{d}, we calculate {d}(ad - bc){d}. Applying that here gives us:

{dd}(4 - \lambda)(4 - \lambda) - (2)(2) = 0{dd}

Now, we expand and simplify the algebra:

{dd}16 - 4\lambda - 4\lambda + \lambda^2 - 4 = 0{dd}
{dd}\lambda^2 - 8\lambda + 12 = 0{dd}

This leaves us with a standard quadratic equation. We can factor this by finding two numbers that multiply to {d}12{d} and add to {d}-8{d}:

{dd}(\lambda - 6)(\lambda - 2) = 0{dd}

Solving for {d}\lambda{d} gives us our two eigenvalues:
* **{d}\lambda_1 = 6{d}**
* **{d}\lambda_2 = 2{d}**


### Step 2: Solving for the Eigenvectors ({d}v{d})

Now that we have our eigenvalues, we plug them one at a time back into the equation {d}(M - \lambda I)v = 0{d} to find the corresponding eigenvector {d}v = \begin{{pmatrix}} v_1 \\ v_2 \end{{pmatrix}}{d}.

#### Finding the first eigenvector (using {d}\lambda_1 = 6{d})

Substitute {d}6{d} for {d}\lambda{d}:

{dd}\begin{{pmatrix}} 4 - 6 & 2 \\ 2 & 4 - 6 \end{{pmatrix}} \begin{{pmatrix}} v_1 \\ v_2 \end{{pmatrix}} = \begin{{pmatrix}} 0 \\ 0 \end{{pmatrix}}{dd}
{dd}\begin{{pmatrix}} -2 & 2 \\ 2 & -2 \end{{pmatrix}} \begin{{pmatrix}} v_1 \\ v_2 \end{{pmatrix}} = \begin{{pmatrix}} 0 \\ 0 \end{{pmatrix}}{dd}

This gives us a system of linear equations:
1.  {d}-2v_1 + 2v_2 = 0{d}
2.  {d}2v_1 - 2v_2 = 0{d}

Both equations simplify to the exact same relationship: {d}v_1 = v_2{d}. This means any vector where the first and second components are equal is an eigenvector for {d}\lambda=6{d}. The simplest whole-number representation is:

{dd}v_{{\lambda1}} = \begin{{pmatrix}} 1 \\ 1 \end{{pmatrix}}{dd}

#### Finding the second eigenvector (using {d}\lambda_2 = 2{d})

Substitute {d}2{d} for {d}\lambda{d}:

{dd}\begin{{pmatrix}} 4 - 2 & 2 \\ 2 & 4 - 2 \end{{pmatrix}} \begin{{pmatrix}} v_1 \\ v_2 \end{{pmatrix}} = \begin{{pmatrix}} 0 \\ 0 \end{{pmatrix}}{dd}
{dd}\begin{{pmatrix}} 2 & 2 \\ 2 & 2 \end{{pmatrix}} \begin{{pmatrix}} v_1 \\ v_2 \end{{pmatrix}} = \begin{{pmatrix}} 0 \\ 0 \end{{pmatrix}}{dd}

This gives us the system:
1.  {d}2v_1 + 2v_2 = 0{d}
2.  {d}2v_1 + 2v_2 = 0{d}

This simplifies to {d}v_1 = -v_2{d}. The simplest whole-number representation here is:

{dd}v_{{\lambda2}} = \begin{{pmatrix}} 1 \\ -1 \end{{pmatrix}}{dd}

### Step 3: Normalizing the Vectors (Crucial for SVD)

In Singular Value Decomposition, the matrix {d}V{d} must be orthogonal, meaning its column vectors need to have a length (magnitude) of exactly 1. 

To normalize a vector, we divide each component by the vector's total length. The length is found using the Pythagorean theorem ({d}\sqrt{{v_1^2 + v_2^2}}{d}).

For our first vector {d}\begin{{pmatrix}} 1 \\ 1 \end{{pmatrix}}{d}, the length is {d}\sqrt{{1^2 + 1^2}} = \sqrt{{2}}{d}. 
* **Normalized {d}v_1{d}:** {d}\begin{{pmatrix}} 1/\sqrt{{2}} \\ 1/\sqrt{{2}} \end{{pmatrix}}{d}

For our second vector {d}\begin{{pmatrix}} 1 \\ -1 \end{{pmatrix}}{d}, the length is {d}\sqrt{{1^2 + (-1)^2}} = \sqrt{{2}}{d}.
* **Normalized {d}v_2{d}:** {d}\begin{{pmatrix}} 1/\sqrt{{2}} \\ -1/\sqrt{{2}} \end{{pmatrix}}{d}

These normalized vectors are the right singular vectors that will become the columns of your {d}V{d} matrix!
"""

    file_name = "README_svd_2.md"
    
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"Successfully created '{file_name}' in the current directory.")
    except Exception as e:
        print(f"Error creating file: {e}")

if __name__ == "__main__":
    generate_readme()
