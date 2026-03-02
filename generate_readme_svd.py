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
"""

    file_name = "README.md"
    
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"Successfully created '{file_name}' in the current directory.")
    except Exception as e:
        print(f"Error creating file: {e}")

if __name__ == "__main__":
    generate_readme()
