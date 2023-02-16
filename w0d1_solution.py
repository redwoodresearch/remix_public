"""
# Pre-MLAB PyTorch Exercises

These exercises will prepare you for actual implementation challenges that will come up during the course. They are not comprehensive, but feeling comfortable with these techniques will save you time during the course and allow you to focus more on concepts and less on coaxing PyTorch and einops to do what you want. 

<!-- toc -->

## Setup

### Installing Python

If you don't have Python 3.8 or higher installed, the most foolproof way is to first install Miniconda from:

[https://docs.conda.io/en/latest/miniconda.htm](https://docs.conda.io/en/latest/miniconda.htm)

### Installing PyTorch

Follow the instructions at [https://pytorch.org](https://pytorch.org/), using the "conda" option if you installed Miniconda. You don't need a GPU or CUDA for these exercises, so you can use the CPU compute platform.

### Installing `einops`

`pip install einops` at the command line should be all you need.

### Visual Studio Code

We will be using [Visual Studio Code](https://code.visualstudio.com/) as the official IDE for the course. Compared to PyCharm, it's free and roughly as good. Compared to Jupyter, it has superior autocomplete, static type-checking, Git integration, and an integrated debugger. Tips for VS Code:

- You can view Markdown files in VS Code by right clicking on the file in the Explorer pane and selecting "Open Preview".
- Enable word wrap in VS Code using Alt+Z or View -> Word Wrap.
- You can have a Jupyter-like notebook experience within a regular `.py` file by using the special comment `# %%` to delimit notebook cells. VS Code will recognize these and give you the option to run cells (Shift-Enter by default on my machine) or debug cells individually.

## Workflow

- The most basic thing you can do is copy the exercises into a new `.py` file and run the file as a normal script. Wherever you see a `pass` statement, this is where you write your code so that the function behaves as described and the assertions pass.
- It's more efficient to use the REPL and split up your `.py` file with notebook cells using `# %%`. Write and check your code one line at a time, verifying they do what you expect and only once you think it's working, paste it into the function body to confirm. This avoids spending time figuring out which line of code isn't working properly.

## Tips

- To get the most out of these exercises, make sure you understand why all of the assertions should be true, and feel free to add more assertions.
- If you're having trouble writing a batched computation, try doing the unbatched version first.
- If you find these exercises challenging, it would be beneficial to go through them a second time so they feel more natural.

## Support

At some point you'll be added to a MLAB Slack and be able to ask questions there. If you need help before then, feel free to email `chris` at `rdwrs.com` directly.
"""
# %%
import math
from einops import rearrange, repeat, reduce
import torch as t


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-5, atol=1e-4) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


# %%
def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    "SOLUTION"
    return rearrange(t.arange(3, 9), "(h w) -> h w", h=3, w=2)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

# %%
def rearrange_2() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    "SOLUTION"
    return rearrange(t.arange(1, 7), "(h w) -> h w", h=2, w=3)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
def rearrange_3() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    "SOLUTION"
    return rearrange(t.arange(1, 7), "a -> 1 a 1")


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))

# %%
"""
## Creating Tensors: Tensor vs tensor

Two ways to create objects of type `t.Tensor` are:

1) Call the constructor of `t.Tensor`
2) Use the creation function `t.tensor`

The constructor way is fraught with peril. What should the following cell print?
"""
# %%
x = t.arange(5)
y = t.Tensor(x.shape)
y2 = t.Tensor(tuple(x.shape))
y3 = t.Tensor(list(x.shape))
print(y, y2, y3)
# %%
"""
What should this cell print?
"""
# %%
x = t.Tensor([False, True])
print(x.dtype)
# %%
"""
Because the first argument can either be interpreted input data OR the shape, and because the constructor silently coerces your input data to the default floating point type, using the constructor is a recipe for bugs. I recommend never calling the constructor, and only using `t.Tensor` for type signatures.

In contrast, `t.tensor` with no dtype specified will try to detect the type of your input automatically. This is usually what you want, but not always. What does the following code do?
"""
# %%
try:
    print(t.tensor([1, 2, 3, 4]).mean())
except Exception as e:
    print("Exception raised: ", e)
# %%

"""
NumPy's `np.mean` would coerce to float and return 2.5 here, but PyTorch detects that your inputs all happen to be integers and refuses to compute the mean because it's ambiguous if you wanted 2.5 or 10 // 4 = 2 instead.

The best practice to avoid surprises and ambiguity is to use `t.tensor` and pass the dtype explicitly. 

Other good ways to create tensors are:

- If you already have a tensor `input` and want a new one of the same size, use functions like [`t.zeros_like(input)`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html). This uses the dtype and device of the input by default, saving you from manually specifying it.
- If you already have a tensor `input` and want a new one with the same dtype and device but new data, use the [`input.new_tensor`](https://pytorch.org/docs/stable/generated/torch.Tensor.new_tensor.html#torch.Tensor.new_tensor) method.
- Many [other creation functions](https://pytorch.org/docs/stable/torch.html#creation-ops) exist, for which you should also specify the dtype explicitly.
"""
# %%
def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    "SOLUTION"
    return reduce(temps, "(h 7) -> h", "mean")


# %%
temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83], dtype=t.float32)
expected = t.tensor([71.5714, 62.1429, 79.0000])
assert_all_close(temperatures_average(temps), expected)

# %%
def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    "SOLUTION"
    avg = repeat(temperatures_average(temps), "w -> (w 7)")
    return temps - avg


expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0000,
        1.0000,
        6.0000,
        1.0000,
        -1.0000,
        -7.0000,
        4.0000,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)
# %%
def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """
    "SOLUTION"
    assert len(temps) % 7 == 0
    avg = repeat(temperatures_average(temps), "w -> (w 7)")
    std = repeat(reduce(temps, "(h 7) -> h", t.std), "w -> (w 7)")
    return (temps - avg) / std


expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)
# %%
def batched_dot_product_nd(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Return the batched dot product of a and b, where the first dimension is the batch dimension.

    That is, out[i] = dot(a[i], b[i]) for i in 0..len(a).
    a and b can have any number of dimensions greater than 1.

    a: shape (b, i_1, i_2, ..., i_n)
    b: shape (b, i_1, i_2, ..., i_n)

    Returns: shape (b, )

    Use torch.einsum. You can use the ellipsis "..." in the einsum formula to represent an arbitrary number of dimensions.
    """
    assert a.shape == b.shape
    "SOLUTION"
    return t.einsum("b...,b... -> b", a, b)


actual = batched_dot_product_nd(t.tensor([[1, 1, 0], [0, 0, 1]]), t.tensor([[1, 1, 0], [1, 1, 0]]))
expected = t.tensor([2, 0])
assert_all_equal(actual, expected)

actual2 = batched_dot_product_nd(t.arange(12).reshape((3, 2, 2)), t.arange(12).reshape((3, 2, 2)))
expected2 = t.tensor([14, 126, 366])
assert_all_equal(actual2, expected2)

# %%
def identity_matrix(n: int) -> t.Tensor:
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    "SOLUTION"
    return (rearrange(t.arange(n), "i->i 1") == t.arange(n)).float()


assert_all_equal(identity_matrix(3), t.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0)))
# %%
def sample_distribution(probs: t.Tensor, n: int) -> t.Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    """
    assert abs(probs.sum() - 1.0) < 1e-3
    assert (probs >= 0).all()
    "SOLUTION"
    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)


n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=1e-3, atol=1e-3)

# %%
def classifier_accuracy(scores: t.Tensor, true_classes: t.Tensor) -> t.Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use torch.argmax.
    """
    assert true_classes.max() < scores.shape[1]
    "SOLUTION"
    return (scores.argmax(dim=1) == true_classes).float().mean()


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected

# %%
def total_price_indexing(prices: t.Tensor, items: t.Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < prices.shape[0]
    "SOLUTION"
    return prices[items].sum().item()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0


# %%
def gather_2d(matrix: t.Tensor, indexes: t.Tensor) -> t.Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    if "SOLUTION":
        assert matrix.ndim == indexes.ndim
        assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    if "SOLUTION":
        assert out.shape == indexes.shape
    return out


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)

indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)


# %%
def total_price_gather(prices: t.Tensor, items: t.Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    "SOLUTION"
    return prices.gather(0, items).sum().item()


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0


# %%
def integer_array_indexing(matrix: t.Tensor, coords: t.Tensor) -> t.Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    "SOLUTION"
    return matrix[tuple(coords.T)]


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))

mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))

# %%
def batched_logsumexp(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    "SOLUTION"
    C = matrix.max(dim=-1).values
    exps = t.exp(matrix - rearrange(C, "n -> n 1"))
    return C + t.log(t.sum(exps, dim=-1))


matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)


# %%
def batched_softmax(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    "SOLUTION"
    exp = matrix.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)

for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))

matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))

# %%
def batched_logsoftmax(matrix: t.Tensor) -> t.Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n). For each i, exp(out[i]) should sum to 1.

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    "SOLUTION"
    C = matrix.max(dim=1, keepdim=True).values
    return matrix - C - (matrix - C).exp().sum(dim=1, keepdim=True).log()


matrix = t.arange(1, 6).view((1, 5)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 6).view((1, 5)).float()

actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]])
assert_all_close(actual, expected)


# %%
def batched_cross_entropy_loss(logits: t.Tensor, true_labels: t.Tensor) -> t.Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    "SOLUTION"
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]

    logprobs = batched_logsoftmax(logits)
    indices = rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    return -rearrange(pred_at_index, "n 1 -> n")


logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)

# %%
def collect_rows(matrix: t.Tensor, row_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    assert row_indexes.max() < matrix.shape[0]
    "SOLUTION"
    return matrix[row_indexes]


matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)

# %%
def collect_columns(matrix: t.Tensor, column_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    "SOLUTION"
    return matrix[:, column_indexes]


matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]])
assert_all_equal(actual, expected)
# %%

"""
## Practice with `torch.as_strided`

If you're not familiar with `as_strided`, a couple good resources are:

- [NumPy For Your Grandma (4 minute video)](https://www.youtube.com/watch?v=VlkzN00P0Bc)
- [as_strided and sum are all you need](https://jott.live/markdown/as_strided)

Fill in the "size" and "stride" arguments so that a call to `as_strided` produces the desired output.

"""
# %%
from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])
test_input_a = t.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])


# %%
if "SOLUTION":
    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]),
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 1, 2], [5, 6, 7]]),
            size=(2, 3),
            stride=(5, 1),
        ),
        TestCase(
            output=t.tensor([[0, 0, 0], [11, 11, 11]]),
            size=(2, 3),
            stride=(11, 0),
        ),
        TestCase(
            output=t.tensor([0, 6, 12, 18]),
            size=(4,),
            stride=(6,),
        ),
        TestCase(
            output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]),
            size=(2, 1, 3),
            stride=(9, 0, 1),
        ),
        TestCase(
            output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
            size=(2, 2, 2, 2),
            stride=(12, 4, 2, 1),
        ),
    ]
else:
    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]),
            size=(1,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 1, 2], [5, 6, 7]]),
            size=(1,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 0, 0], [11, 11, 11]]),
            size=(1,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([0, 6, 12, 18]),
            size=(1,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]),
            size=(1,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
            size=(1,),
            stride=(1,),
        ),
    ]

for i, case in enumerate(test_cases):
    actual = test_input_a.as_strided(size=case.size, stride=case.stride)
    if (case.output != actual).any():
        print(f"Test {i} failed:")
        print(f"Expected: {case.output}")
        print(f"Actual: {actual}")
    else:
        print(f"Test {i} passed!")

"""
## Implementing ReLU

Implement ReLU five different ways. 
"""

# %%
def test_relu(relu_func):
    print(f"Testing: {relu_func.__name__}")
    x = t.arange(-1, 3, dtype=t.float32, requires_grad=True)
    out = relu_func(x)
    expected = t.tensor([0.0, 0.0, 1.0, 2.0])
    assert_all_close(out, expected)


def relu_clone_setitem(x: t.Tensor) -> t.Tensor:
    """Make a copy with torch.clone and then assign to parts of the copy."""
    "SOLUTION"
    x = x.clone()
    x[x < 0.0] = 0.0
    return x


test_relu(relu_clone_setitem)

# %%
def relu_where(x: t.Tensor) -> t.Tensor:
    """Use torch.where."""
    "SOLUTION"
    return t.where(x > 0.0, x, t.tensor(0.0))


test_relu(relu_where)

# %%
def relu_maximum(x: t.Tensor) -> t.Tensor:
    """Use torch.maximum."""
    "SOLUTION"
    return t.maximum(x, t.tensor(0.0))


test_relu(relu_maximum)

# %%
def relu_abs(x: t.Tensor) -> t.Tensor:
    """Use torch.abs."""
    "SOLUTION"
    return (x.abs() + x) / 2.0


test_relu(relu_abs)

# %%
def relu_multiply_bool(x: t.Tensor) -> t.Tensor:
    """Create a boolean tensor and multiply the input by it elementwise."""
    "SOLUTION"
    return x * (x > 0)


test_relu(relu_multiply_bool)

# %%
