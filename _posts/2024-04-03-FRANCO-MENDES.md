---
title: "FRANCO-MENDES"
date: 2024-04-03
---

# Introduction to Matrices
First we start with basic definitions of matrices and tensors. 
If you are reading this article you probably know what a matrix is, but here is one anyway.

$$
B = \begin{bmatrix}
1.23 & 2.45 & 3.67 & 4.89 \\
5.01 & 6.32 & 7.54 & 8.76 \\
9.87 & 10.98 & 11.21 & 12.34
\end{bmatrix}
$$

Again, consider  a matrix multiplication with a vector, 

$$
R = \begin{bmatrix} 1.23 & 2.45 & 3.67 & 4.89 \\ 
5.01 & 6.32 & 7.54 & 8.76 \\ 
9.87 & 10.98 & 11.21 & 12.34 \end{bmatrix} \begin{bmatrix} 
2 \\ 
3 \\ 
4 
\end{bmatrix} 
$$

# Size
Given the matrix above the main focus of this article will be to reduce the matrix size so that we never actually store all 12 elements of the matrix. With that said here is a more precise definition of how much space the matrix above takes on hardware memory. 
In IEEE 754 single-precision format:

$$
\begin{itemize}
  \item 1 bit is used for the sign (positive or negative).
  \item 8 bits are used for the exponent.
  \item 23 bits are used for the significand (also called mantissa or fraction).
\end{itemize}
$$


The total of 32 bits (or 4 bytes) is used to represent a single floating-point number.
 So for this above matrix we need 12 memory locations or $4 x 12$ bytes. So if you were to save this matrix on a piece of hardware it would take up 4 bytes. The same applies for a tensor with 12 elements, which looks like this,

$$
T = \begin{bmatrix}
    \begin{bmatrix}
        11.23 & 2.34 \\
        3.45 & 4.56 \\
        5.67 & 16.78 \\
    \end{bmatrix} &
    \begin{bmatrix}
        7.89 & 8.91 \\
        9.12 & 10.23 \\
        11.34 & 12.45 \\
    \end{bmatrix} \\
\end{bmatrix}
$$

While it is useful to think of tensors as a list of matrices, it is important to note that they have some important differences as mathematical objects. It is perhaps more useful to think of matrices as a "special case" of a tensor. 


# Number of Operations
For the given operation, we're performing a matrix multiplication of a 3×43×4 matrix with a 4×14×1 column vector. The resulting matrix will be a 3×13×1 column vector.

To compute each element of the resulting vector, we perform the dot product of each row of the matrix with the column vector. Since each row has 4 elements, this involves 4 multiplications and 3 additions.

Therefore, for the entire operation:

$$
\begin{itemize}
    \item There are 3 rows in the matrix.
    \item For each row, there are 4 multiplications and 3 additions.
\end{itemize}
$$

Hence, the total number of multiplications is 3×4=123×4=12, and the total number of additions is 3×3=93×3=9.

So, in total, there are 12 multiplications and 9 additions involved in the matrix-vector multiplication operation. For two matrices being multiplied of dimensions $m\times k$ and $k\times n$, there are $m\times n \times k$ multiplications. And the number of additions is $m\times n \times (k-1) $. Notice that the number of multiplies is always greater than the number of sums. 

# Neural Networks as sequences of matrix/ tensor operations

A Neural Network is simply a sequence of tensor operations. In this section we will outline a simple neural network and illustrate this concept. 
Input (3 nodes) --> Hidden Layer (2 nodes, ReLU) --> Output Layer (2 nodes, Sigmoid)

$$
X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
,
W^{(1)} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{bmatrix}
,
b^{(1)} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
,
W^{(2)} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}
,
b^{(2)} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$

Hidden layer:

$$
Z^{(1)} = W^{(1)} X + b^{(1)}
$$


$$
A^{(1)} = \text{ReLU}(Z^{(1)})
$$

Output layer:
$$
Z^{(2)} = W^{(2)} A^{(1)} + b^{(2)}
$$

$$
A^{(2)} = \text{sigmoid}(Z^{(2)})
$$

We can combine these equations into one :
$$A^{(2)} = \text{sigmoid}(W^{(2)} \text{ReLU}(W^{(1)} X + b^{(1)}) + b^{(2)}) $$

In embedded systems its common to just feed an extra input of 1's to the input and drop the biases. If you are familiar with matrix formulations of linear regression, you are probably familiar with this, but if not, you can see this clearly by the following,
$$
W'X = \begin{bmatrix} w_{11} & w_{12} & b_1 \\ w_{21} & w_{22} & b_2 \end{bmatrix} \begin{bmatrix} x_1 & 1 \\ x_2 & 1 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + b_1 \\ w_{21}x_1 + w_{22}x_2 + b_2 \end{bmatrix}
$$

Thus, after adding in the 1s to X and the column of biases to $W$ we get, 

$$ A^{(2)} = \text{sigmoid}(W^{(2)} \text{ReLU}(W^{(1)} X)) $$
This X and W are modified versions of the X, W mentioned before but we will stick to this notation for the rest of this article.  You can generalize this to as many layers as you want. 
$$ \hat{y} = A^{(L)} = \text{activation}(W^{(L)} \text{activation}(W^{(L-1)} \dots \text{activation}(W^{(1)} X))) $$

$$ \hat{y} = A^{(L)} = \text{activation}(W^{(L)} \text{activation}(W^{(L-1)} \dots \text{activation}(USV' X))) $$







```python
print("hello")
def e(c,m):
  return c*m**2
```




