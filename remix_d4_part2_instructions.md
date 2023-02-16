
# Day 4b: Paren Balancer Causal Scrubbing

To start, please read this [LessWrong post](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y/p/kjudfaQazMmC74SbF).

We will replicate the experiments today.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Setup](#setup)
    - [Circuit loading](#circuit-loading)
    - [dataset and experiment code](#dataset-and-experiment-code)
    - [Helpful tidbit on tests:](#helpful-tidbit-on-tests)
- [Experiment 0: Baselines](#experiment--baselines)
- [Experiment 1: The direct contributions to the output](#experiment--the-direct-contributions-to-the-output)
    - [Matchers](#matchers)
    - [Cond Samplers](#cond-samplers)
- [Experiment 2: Diving into heads 1.0 and 2.0](#experiment--diving-into-heads--and-)
    - [Part 1: Splitting up the input to 1.0 and 2.0 by sequence position](#part--splitting-up-the-input-to--and--by-sequence-position)
    - [Part 2](#part-)
    - [Part 3: Projecting 0.0 onto a single direction](#part--projecting--onto-a-single-direction)
        - [Circuit rewrite](#circuit-rewrite)
    - [Part 4: The $\phi$ function](#part--the-phi-function)
- [Experiment 3: Head 2.1](#experiment--head-)
- [Experiment 4: All together](#experiment--all-together)

## Learning Objectives

After today's material, you should be able to:

- Execute causal scrubbing experiments using the `causal_scrubbing` package.
- Apply more complex rewrites in Circuits.

Note: unlike Circuits which is in Rust, `causal_scrubbing` is in Python so feel free to use Go To Definition and look at the implementation to clarify what's going on.


```python
import os
import sys
import uuid
from typing import Optional, Tuple
import rust_circuit.optional as op
import numpy as np
import rust_circuit as rc
import torch
from rust_circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    CondSampler,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from rust_circuit.algebric_rewrite import residual_rewrite, split_to_concat
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_model_id
from rust_circuit.py_utils import I
from torch.nn.functional import binary_cross_entropy_with_logits
import remix_d4_part2_test as tests
from remix_d4_part2_setup import ParenDataset, ParenTokenizer, get_h00_open_vector

MAIN = __name__ == "__main__"
SEQ_LEN = 42
NUM_EXAMPLES = 4000
MODEL_ID = "jun9_paren_balancer"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
DEFAULT_CHECKS: ExperimentCheck = True
EVAL_DEVICE = "cpu"
MAX_MEMORY = 20000000000
BATCH_SIZE = 2000

```

## Setup
No exercises here! It may be helpful to read over the code, however.

### Circuit loading
If any of these operations confuse you, try printing out the circuit before and after!

Step 1: Initial loading


```python
circ_dict, _, model_info = load_model_id(MODEL_ID)
circuit = circ_dict["t.bind_w"]

```

Step 2: We bind the model to an input by attaching a placeholder symbol input named "tokens" to the model. We then specify the attention mask that prevents attending to padding depends on this tokens array.

We use one hot tokens as this makes defining the attention mask a simple indexing operation.

The symbol has a random fixed uuid (fixed as this gives consistency when comparing in tests).


```python
toks_uuid = uuid.UUID("ce34280e-169f-40bd-b78e-8adeb4274aba")
tokens_arr = rc.Symbol((SEQ_LEN, ParenTokenizer.vocab_size), uuid=toks_uuid, name="tokens")
tok_embeds = rc.Einsum.from_fancy_string(
    "seqlen vocab_size, vocab_size hidden -> seqlen hidden", tokens_arr, circ_dict["t.w.tok_embeds"], name=f"tok_embeds"
)
attn_mask = rc.Add.minus(rc.Scalar(1), rc.Index(tokens_arr, I[:, ParenTokenizer.PAD_TOKEN]), name="pos_mask")
circuit = model_info.bind_to_input(circuit, tok_embeds, circ_dict["t.w.pos_embeds"], attn_mask)

```

Step 3: rewrites the circuit into a more convenient structure to work with using `configure_transformer`.
This flattens out the residual stream (as opposed to the nested layer structure originally) and, pushes down the weight bindings, and separates out each attention layer into a sum of heads.


```python
circuit = circuit.update(
    "t.bind_w",
    lambda c: configure_transformer(
        c,
        To.ATTN_HEAD_MLP_NORM,
        split_by_head_config="full",
        use_pull_up_head_split=True,
        use_flatten_res=True,
        flatten_components=True,
    ),
)

```

Some additional misc rewrites.

We substitute the inputs to be duplicated everywhere they apperar in the model instead of being in one outer module bind.

We also index as we only care about the classification at position 0, and use `rc.conform_all_modules` to replace any remaining symbolic shapes with their numeric values.


```python
circuit = circuit.cast_module().substitute()
circuit = rc.Index(circuit, I[0]).rename("logits_pos0")
circuit = rc.conform_all_modules(circuit)

```

Finally, some custom renames that make the circuit more intuitive.


```python
circuit = circuit.update("t.call", lambda c: c.rename("logits"))
circuit = circuit.update("t.call", lambda c: c.rename("logits_with_bias"))
circuit = circuit.update(rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner"))
circuit = circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
circuit = circuit.update("t.a.mask", lambda c: c.rename("padding_mask"))
for l in range(model_info.params.num_layers):
    circuit = circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
    circuit = circuit.update(f"b{l}.a.h0", lambda c: c.rename(f"a{l}.h0"))
    circuit = circuit.update(f"b{l}.a.h1", lambda c: c.rename(f"a{l}.h1"))
    next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
    circuit = circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))
printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    traversal=rc.restrict(
        rc.IterativeMatcher("embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")), term_if_matches=True
    ),
)
if PRINT_CIRCUITS:
    printer.print(circuit)
circuit = rc.cast_circuit(circuit, rc.TorchDeviceDtypeOp(device=EVAL_DEVICE))

```

### dataset and experiment code
We have a custom dataset class that precomputes some features of paren sequences, and handles pretty printing / etc.


```python
ds = ParenDataset.load(device=EVAL_DEVICE)


def bce_with_logits_loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    Computes the binary cross entropy loss for the provided labels.
    logits: [batch, 2]. Class 0 is unbalanced logit, class 1 is balanced logit.
    labels: [batch]. True if balanced.
    """
    targets = labels.to(dtype=logits.dtype, device=logits.device)
    logit_diff = logits[..., 1] - logits[..., 0]
    correct = (logit_diff > 0) == targets
    loss = binary_cross_entropy_with_logits(logit_diff, targets, reduction="none")
    return (loss, correct)


def paren_experiment(
    circuit: rc.Circuit,
    dataset: ParenDataset,
    corr: Correspondence,
    checks: ExperimentCheck = DEFAULT_CHECKS,
    random_seed=1,
    actually_run=ACTUALLY_RUN,
    num_examples=NUM_EXAMPLES,
    batch_size=BATCH_SIZE,
    **kwargs,
) -> Tuple[ScrubbedExperiment, Optional[float]]:
    ex = Experiment(circuit, dataset, corr, random_seed=random_seed, check=checks, **kwargs)
    scrubbed = ex.scrub(num_examples, treeify=actually_run)
    overall_loss: Optional[float] = None
    if actually_run:
        logits = scrubbed.evaluate(
            ExperimentEvalSettings(
                optim_settings=rc.OptimizationSettings(max_memory=MAX_MEMORY, scheduling_naive=True),
                device_dtype=EVAL_DEVICE,
                optimize=True,
                batch_size=batch_size,
            )
        )
        ref_ds = ParenDataset.unwrap(scrubbed.ref_ds)
        labels = ref_ds.is_balanced.value

        def loss_str(mask):
            loss, correct = bce_with_logits_loss(logits[mask], labels[mask])
            loss = loss.cpu()
            std_err = loss.std() / len(loss) ** 0.5
            return f"{loss.mean():.3f}  SE={std_err:.3f}  acc={correct.float().mean():.1%} "

        print(f"  overall:               {loss_str(slice(None))}")
        print(f"    on bal:              {loss_str(labels.to(dtype=torch.bool))}")
        print(f"    on unbal:            {loss_str(~labels.to(dtype=torch.bool))}")
        print(f"    on count failures:   {loss_str(~ref_ds.count_test.to(dtype=torch.bool))}")
        print(f"    on horizon failures: {loss_str(~ref_ds.horizon_test.to(dtype=torch.bool))}")
        overall_loss = bce_with_logits_loss(logits, labels)[0].mean().item()
    return (scrubbed, overall_loss)


def check_loss(loss: Optional[float], target: float, std_err: float):
    assert loss is not None
    err = abs(loss - target)
    assert err < 4 * std_err, f"err too large! loss ({loss:.2f}) != target ({target:.2f}) ± 4*SE ({std_err:.2f})"
    if err > 2 * std_err:
        raise Warning(f"Err is kinda large! loss ({loss:.2f}) != target ({target:.2f}) ± 2*SE ({std_err:.2f})")

```

### Helpful tidbit on tests:
Most of the tests in this file raise assertion errors that contain extra data, for instance the objects from the comparison that failed. It can be convenient to catch this data to debug. For instance:

```
def check_eq(a, b):
    assert a == b, ("not equal!", a, b)

try:
    check_eq(0, 1)
except AssertionError as e:
    a, b = e.args[0][1], e.args[0][2]
    print(a, b)
```


## Experiment 0: Baselines
To start with let's measure two baselines:
  - running the model normally
  - interchanging the logits randomly

Make causal scrubbing experiments that impliment both of these. In each case there should be a single interp node named "logits".

The tests do explictly check that the interp nodes in the correspondence are named correctly, in order to facilitate more helpful feedback.


```python
corr0a = Correspondence()
"TODO: Your code here"
if MAIN:
    tests.t_ex0a_corr(corr0a)
    print("\nEx0a: Exact sampling")
    ex0a, loss0a = paren_experiment(circuit, ds, corr0a)
    check_loss(loss0a, 0, 0.01)
corr0b = Correspondence()
"TODO: Your code here"
if MAIN:
    tests.t_ex0b_corr(corr0b)
    print("\nEx0b: Interchanging logits")
    ex0b, loss0b = paren_experiment(circuit, ds, corr0b)
    check_loss(loss0b, 4.3, 0.12)

```

## Experiment 1: The direct contributions to the output
Now, let's construct a basic experiment to determine the role that different heads play.
We'll start by testing the following claimed hypothesis:
 - Heads 1.0 and 2.0 compute the count test, and check that there are equal numbers of open and close parentheses
 - Head 2.1 computes the horizon test.

### Matchers

Define the following matchers. You only want to match _direct_ paths, that is paths through the residual stream and not through indirect paths. This can be accomplished by calling `rc.restrict` or the `chain_excluding` utilty included in causal scrubbing code.



```python
m_10: rc.IterativeMatcher
m_20: rc.IterativeMatcher
m_21: rc.IterativeMatcher
"TODO: Your code here"
if MAIN:
    tests.t_m_10(m_10)
    tests.t_m_20(m_20)
    tests.t_m_21(m_21)

```

### Cond Samplers

Let's now define some samplers that sample according to whether or not a datum passes the count test and the horizon test. We first need to define some helper functions, then wrap them in `FuncSampler`s. You will need to use `ParenDataset.tokens_flat`.

Note: these are also predefined properties on `ParenDataset`. If you are short on time, feel free to use them and skip this exercise. The version inside `ParenDataset` also uses caching to make things faster, so if speed gets to be annoyingly slow for later experiments you could switch over).

<details>
<summary>
Using functions that come with `ParenDataset`
</summary>

You can use the utilities in `ParenDataset` by changing your code to the following:
```
count_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).count_test)

horizon_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).horizon_test)
```
</details>


```python
def passes_count(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)]
    Result is true when the corresponding datum has equal numbers of open and close parens
    """
    "TODO: Your code here"
    raise NotImplementedError


def passes_horizon(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)]
    Result is true when the corresponding datum passes the right to left horizon test as described in the [writeup](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y/p/kjudfaQazMmC74SbF#Algorithm).
    """
    "TODO: Your code here"
    raise NotImplementedError


count_cond = FuncSampler(lambda d: passes_count(ParenDataset.unwrap(d)))
horizon_cond = FuncSampler(lambda d: passes_horizon(ParenDataset.unwrap(d)))
if MAIN:
    tests.t_count_cond(count_cond)
    tests.t_horizon_cond(horizon_cond)

```

This first correspondence should have 4 nodes:
 - The root node, named "logits", with an ExactSampler (any sampler that agrees on the labels will be equivalent,
 but an exact sampler is somewhat more computationally efficient).
 - Three nodes for the three heads of interest, named "10", "20", and "21". The first two should use the cond sampler provided (`cs_for_h10_and_h20`, which will be `count_cond` for now), the third should use the `horizon_cond` sampler.

The exact interp node names are checked by the test, which allows it to give more meaningful feedback.


```python
def make_ex1_corr(cs_for_h10_and_h20: CondSampler) -> Correspondence:
    """TODO: YOUR CODE HERE"""
    pass


if MAIN:
    print("\nEx1a: Just the count cond")
    tests.t_ex1_corr(make_ex1_corr, count_cond)
    ex1a, loss1a = paren_experiment(circuit, ds, make_ex1_corr(count_cond))
    check_loss(loss1a, 0.52, 0.04)
    if PRINT_CIRCUITS:
        ex1a.print()

```

As discussed in the writeup, we can more accurately capture the equivalence classes of 1.0 and 2.0's output by including if the first parenthesis is open or closed.

This is a natural feature for these heads to use: a sequence will always be unbalanced if it starts with a close parenthesis, and as these heads depend strongly on the residual stream at position 1 anyhow (as we will show in experiement 2) the information is readily accessible.

(reminder: position 0 is always the [START] token, so position 1 is the first parentheses. All sequences in our dataset have >= 2 parentheses in them, so you can assume position 1 is either an open or close paren.)

Define some new cond samplers that incorporate this feature. The first one checks that the first paren is an open parentheses, and the next one tests the input passes count test AND the first paren is open.
We won't use the pure `start_open` test quite yet, but we will soon and it's nice to define it here.

Again, click on the hint below if you are short on time and just want to use the versions native to the `ParenDataset` class.

<details>
<summary>
Using functions that come with `ParenDataset`
</summary>

You can use the utilities in `ParenDataset` by changing your code to the following:
```
start_open_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).starts_with_open)

count_open_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).count_test & ParenDataset.unwrap(d).starts_with_open)
```
</details>


```python
def passes_starts_open(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)].
    Result is true when the corresponding datum starts with '('.
    """
    "TODO: Your code here"
    raise NotImplementedError


def passes_count_open(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)].
    Result is true when the corresponding datum starts with '(' and there are equal numbers of open and close parens in the entire sequence.
    """
    "TODO: Your code here"
    raise NotImplementedError


start_open_cond = FuncSampler(lambda d: passes_starts_open(ParenDataset.unwrap(d)))
count_open_cond = FuncSampler(lambda d: passes_count_open(ParenDataset.unwrap(d)))
if MAIN:
    tests.t_start_open_cond(start_open_cond)
    tests.t_count_open_cond(count_open_cond)
if MAIN:
    print("\nEx1b: Without a0")
    tests.t_ex1_corr(make_ex1_corr, count_open_cond)
    ex1b, loss1b = paren_experiment(circuit, ds, make_ex1_corr(count_open_cond))
    check_loss(loss1b, 0.3, 0.04)
    if PRINT_CIRCUITS:
        ex1b.print()

```

Bonus: Can you improve on the loss by specifying other direct paths, or choosing better features to ensure
agreement along?


```python
"TODO: YOUR CODE HERE"

```

## Experiment 2: Diving into heads 1.0 and 2.0
We are going split up experiment 2 into four parts:
 - Part 1: 1.0 and 2.0 only depend on their input at position 1
 - Part 2 (ex2a in writeup): 1.0 and 2.0 only depend on:
    - the output of 0.0 (which computes $p$, the proportion of open parentheses) and
    - the embeds (which encode if the first paren is open)
 - Part 3: Projecting the output of 0.0 onto a single direction
 - Part 4 (ex2b in writeup): Estimate the output of 0.0 with a function $\phi(p)$

### Part 1: Splitting up the input to 1.0 and 2.0 by sequence position
One of the claims we'd like to test is that only the input at position 1 (the first paren position) matters for both heads 1.0 and 2.0.
Currently, however, there is no node of our circuit corresponding to "the input at position 1". Let's change that!

Write a `separate_pos1` function that will transform a circuit `node` into:
```
'node_concat' Concat
  'node.at_pos0' Index [0:1, :]
    'node'
  'node.at_pos1' Index [1:2, :]
    'node'
  'node.at_pos2_42' Index [2:42, :]
    'node'
```
This can be acomplished by calling `split_to_concat()` (from algebraic_rewrites.py) and `.rename`-ing the result. If your test is not passing, take a look at your circuit and see how `split_to_concat()` names things by default.


```python
def separate_pos1(c: rc.Circuit) -> rc.Circuit:
    """TODO: YOUR CODE HERE"""
    pass

```

Then split the input node, but only along paths that are reached through head 2.0 and 1.0. (We don't want to split the input to 2.1 in particular, as we'll split that differently later.)

You'll probably want to do this with two different `.update` calls. One that splits the input to 2.0 and one that splits the input to 1.0.

If you are struggling to get the exact correct circuit, you are free to import it from the solution file and print it out. You can also try `print(rc.diff_circuits(your_circuit, our_circuit))` though.

Yes, the tests are very strict about naming things exactly correctly.
This is partially because it is convenient for tests, but also because names are really important!
Good names reduce confusion about what that random node of the circuit actually means.
Mis-naming nodes is also a frequent cause of bugs, e.g. a matcher that traverses a path that it wasn't supposed to.

The tests are also, unfortunately strict about the exact way that the idexes are specified.
This is because `rc.Index(arr, 3:8) != rc.Index(arr, t.arange(3,8))` even though they are functionally equivilant.


```python
ex2_part1_circuit = circuit
"TODO: YOUR CODE HERE"
if MAIN and PRINT_CIRCUITS:
    subcirc = ex2_part1_circuit.get_unique(rc.IterativeMatcher("a2.h0").chain("a2.input_concat"))
    printer.print(subcirc)
if MAIN:
    tests.t_ex2_part1_circuit(ex2_part1_circuit)

```

Now we can test the claim that both 1.0 and 2.0 only cares about position 1!

We'll need new matchers, which just match the pos1 input.


```python
m_10_p1: rc.IterativeMatcher
m_20_p1: rc.IterativeMatcher
"TODO: Your code here"
if MAIN:
    tests.t_m_10_p1(m_10_p1)
    tests.t_m_20_p1(m_20_p1)

```

Then create a correspondence that extends the one returned by `make_ex1_corr(count_open_cond)` so that both 1.0 and 2.0 only use information from position 1. `Correspondence.get_by_name` is useful here.

Have your new nodes be named "10_p1" and "20_p1".


```python
def make_ex2_part1_corr() -> Correspondence:
    """TODO: YOUR CODE HERE"""
    pass


if MAIN:
    tests.t_make_ex2_part1_corr(make_ex2_part1_corr())
    print("\nEx 2 part 1: 1.0/2.0 depend on position 1 input")
    ex2_p1, loss2_p1 = paren_experiment(ex2_part1_circuit, ds, make_ex2_part1_corr())

```

### Part 2
We now construct experiment 2a from the writeup. We will be strict about where 1.0 and 2.0 learn the features they depend on. We claim that the 'count test' is determined by head 0.0 checking the exact proportion of open parens in the sequence and outputting this into the residual stream at position 1.

We thus need to also split up the output of attention head 0.0, so we can specify it only cares about the output of this head at position 1. Again, let's only split it for the branch of the circuit we are working with: copies of 0.0 that are upstream of either `m_10_p1` or `m_20_p1`.


```python
ex2_part2_circuit = ex2_part1_circuit
"TODO: YOUR CODE HERE"
if MAIN and PRINT_CIRCUITS:
    printer.print(ex2_part2_circuit.get_unique(m_10_p1))
if MAIN:
    tests.t_ex2_part2_circuit(ex2_part2_circuit)

```

First, make a new cond sampler that samples an input that agrees on what is called $p_1^($ in the writeup. This can be done with a `FuncSampler` based on a function with the following equivalence classes:
 - one class for _all_ inputs that start with a close parenthesis
 - one class for every value of $p$ (proportion of open parentheses in the entire sequence)

Note the actual values returned aren't important, just the equivalence clases. Also, remember to check the `ParenDataset` class for useful properties. For example, you might consider using `d.starts_with_open` and `d.p_open_after`.


```python
def p1_if_starts_open(d: ParenDataset):
    """Returns a tensor of size [len(ds)]. The value represents p_1 if the sequence starts open, and is constant otherwise"""
    "TODO: YOUR CODE HERE"
    pass


p1_open_cond = FuncSampler(lambda d: p1_if_starts_open(ParenDataset.unwrap(d)))

```
And some matchers


```python
m_10_p1_h00: rc.IterativeMatcher
m_20_p1_h00: rc.IterativeMatcher
"TODO: Your code here"

```

Now make the correspondence!

You should add 4 nodes to the correspondence from part 1:
 - "10_p1_00"
 - "20_p1_00"
 - "10_p1_emb"
 - "20_p1_emb"

Refer to the writeup if you are unsure what to do with the `emb` nodes. This is [Experiment 2a](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y/p/kjudfaQazMmC74SbF#2a__Dependency_on_0_0) in the writeup.


```python
def make_ex2_part2_corr() -> Correspondence:
    """TODO: YOUR CODE HERE"""
    pass


if MAIN:
    tests.t_ex2_part2_corr(make_ex2_part2_corr())
    print("\nEx 2 part 2 (2a in writeup): 1.0/2.0 depend on position 0.0 and emb")
    ex2a, loss2a = paren_experiment(ex2_part2_circuit, ds, make_ex2_part2_corr())
    check_loss(loss2a, 0.55, 0.04)

```

### Part 3: Projecting 0.0 onto a single direction
#### Circuit rewrite
Another claim we would like to test is that only the output of 0.0 written in a particular direction is important.

To do this we will rewrite the output of 0.0 as the sum of two terms: the [projection](https://en.wikipedia.org/wiki/Vector_projection) and rejection (aka the perpendicular component) along this direction.


```python
h00_open_vector = get_h00_open_vector(MODEL_ID)


def project_into_direction(c: rc.Circuit, v: torch.Tensor = h00_open_vector) -> rc.Circuit:
    """
    Rename `c` to `f"{c.name}_orig"`.
    Then return a circuit that computes (the renamed) `c`: [seq_len, 56] projected onto the direction of vector `v`: [56].
    Call the resulting circuit `{c.name}_projected`.
    """
    "TODO: YOUR CODE HERE"
    pass


if MAIN:
    tests.t_project_into_direction(project_into_direction)


def get_ex2_part3_circuit(c: rc.Circuit, project_fn=project_into_direction):
    """
    Uses `residual_rewrite` to write head 0.0 at position 1 (when reached by either `m_10_p1_h00` or `m_20_p1_h00`), as a sum of the projection and the rejection along h00_open_vector. The head retains its same name, with children named `{head.name}_projected` and `{head.name}_projected_residual`.
    """
    "TODO: YOUR CODE HERE"
    pass


ex2_part3_circuit = get_ex2_part3_circuit(ex2_part2_circuit)
if MAIN and PRINT_CIRCUITS:
    proj_printer = printer.evolve(
        traversal=rc.new_traversal(term_early_at={"a0.h0.at_pos0", "a0.h0.at_pos2_42", "a0.h0_orig"})
    )
    subcirc = ex2_part3_circuit.get_unique(m_10_p1.chain("a0.h0_concat"))
    proj_printer.print(subcirc)
if MAIN:
    tests.t_ex2_part3_circuit(get_ex2_part3_circuit)

```

Now make the correspondence. Be sure to avoid the residual node!

This correspondence requires adding two new nodes:
 - "10_p1_00_projected"
 - "20_p1_00_projected"


```python
def make_ex2_part3_corr() -> Correspondence:
    """TODO: YOUR CODE HERE"""
    pass


if MAIN:
    tests.t_ex2_part3_corr(make_ex2_part3_corr())
    print("\nEx 2 part 3: Projecting h00 into one direction")
    ex2_p3, loss2_p3 = paren_experiment(ex2_part3_circuit, ds, make_ex2_part3_corr())

```

### Part 4: The $\phi$ function


```python
def compute_phi_circuit(tokens: rc.Circuit):
    """
    tokens: [seq_len, vocab_size] array of one hot tokens representing a sequence of parens
    (see ParenTokenizer for the one_hot ordering)

    Returns a circuit that computes phi: tokens -> R^56
    phi = h00_open_vector(2p - 1)
    where p = proportion of parens in `tokens` that are open.

    Returns a circuit with name 'a0.h0_phi'.
    """
    "TODO: YOUR CODE HERE"
    pass


if MAIN:
    tests.t_compute_phi_circuit(compute_phi_circuit)


def get_ex2_part4_circuit(orig_circuit: rc.Circuit = ex2_part2_circuit, compute_phi_circuit_fn=compute_phi_circuit):
    """
    Split the output of head 0.0 at position 1, when reached through the appropriate paths, into a phi estimate
    and the residual of this estimate.

    The resulting subcircuit should have name 'a0.h0' with children 'a0.h0_phi' and 'a0.h0_phi_residual'.
    """
    "TODO: YOUR CODE HERE"
    pass


ex2_part4_circuit = get_ex2_part4_circuit()
if MAIN and PRINT_CIRCUITS:
    proj_printer = printer.evolve(
        traversal=rc.new_traversal(term_early_at={"a0.h0.at_pos0", "a0.h0.at_pos2_42", "a0.h0_orig"})
    )
    subcirc = ex2_part4_circuit.get_unique(m_10_p1.chain("a0.h0_concat"))
    proj_printer.print(subcirc)
if MAIN:
    tests.t_ex2_part4_circuit(get_ex2_part4_circuit)

```

And now make the correspondence -- it should be very similar to the one from part 3. Build on top of the one from part **2**, with new node names "10_p1_00_phi" and "20_p1_00_phi".


```python
def make_ex2_part4_corr() -> Correspondence:
    """TODO: YOUR CODE HERE"""
    pass


if MAIN:
    tests.t_ex2_part4_corr(make_ex2_part4_corr())
    print("Ex2 part 4 (2b in writeup): replace a0 by phi(p)")
    ex2b, loss2b = paren_experiment(ex2_part4_circuit, ds, make_ex2_part4_corr())
    check_loss(loss2b, 0.53, 0.04)

```

Congradulations! This is the end of the main part of today's content. Below is some additional content that covers experiments 3 and 4 from the writeup, although with less detailed testing and instructions.


## Experiment 3: Head 2.1

This experiment is officially optional content, and has less instructions / tests along the way. I expect it to be valuable practice for expressing informal hypotheses within the framework, but it may be somewhat more difficult to figure out exactly what the tests expect!


```python
def separate_all_seqpos(c: rc.Circuit) -> rc.Circuit:
    """
    Separate c into all possible sequence positions.
    c is renamed to `{c.name}_concat`, with children `{c.name}.at_pos{i}`
    """
    "TODO: YOUR CODE HERE"
    pass


if MAIN:
    tests.t_separate_all_seqpos(separate_all_seqpos)
ex3_circuit = circuit
"TODO: YOUR CODE HERE"
if MAIN and PRINT_CIRCUITS:
    printer.print(ex3_circuit.get_unique(rc.IterativeMatcher("a2.input_concat")))
if MAIN:
    tests.t_ex3_circuit(ex3_circuit)

```

When adjusted = True, use the `ds.adjusted_p_open_after` attribute instead of `ds.p_open_after` to compute the horizon test.

One possible gotcha in this section is late-binding-closures messing with the values of i. I think if you follow the outline you should be fine, but if you get strange bugs it's one possibility.


```python
def to_horizon_vals(d: ParenDataset, i: int, adjusted: bool = False) -> torch.Tensor:
    """
    Returns a value for the horizon_i test dividing up the input datums into 5 equivalence classes.
    The actual numerical return values don't have inherent meaning, but are defined as follows:
        0 on padding,
        positive on plausibly-balanced positions,
        negative on unbalance-evidence positions,
        1 / -1 on END_TOKENS,
        2 / -2 on non-end tokens
    """
    "TODO: YOUR CODE HERE"
    pass


if MAIN:
    tests.t_to_horizon_vals(to_horizon_vals)


def get_horizon_cond(i: int, adjusted: bool) -> FuncSampler:
    """Func sampler for horizon_i"""
    "TODO: YOUR CODE HERE"
    pass


def get_horizon_all_cond(adjusted: bool) -> FuncSampler:
    """Func sampler horizon_all"""
    "TODO: YOUR CODE HERE"
    pass


if MAIN:
    tests.t_get_horizon_all_cond(get_horizon_all_cond)


def make_ex3_corr(adjusted: bool = False, corr=None) -> Correspondence:
    """
    `adjusted`: uses `adjusted_p_open_after` based conditions if True, `p_open_after` otherwise.
    `corr`: The starting corr. Uses experiemnt 1b corr by default (that is `make_ex1_corr(count_open_cond)`).

    Makes the following modifications:
     - Changes the cond sampler on node `21` to be the horizon_all cond sampler.
     - Adds one node for each sequence position, called `21_p{i}` with horizon_i cond sampler.
     - Also adds a node `pos_mask`, ensuring the pos_mask of head 2.1 is sampled from an input with the same input length.
    """
    "TODO: YOUR CODE HERE"
    pass

```

Note, we have a mini-replication crisis and with the current code can't replicate the exact numbers from the writeup. The handling of the position mask is somewhat different, although much more sane imo. I haven't had time to diagnose the exact cause of the difference.

In any case, expect your loss to be closer to ~1.23 for ex3a and ~1.17 for ex3b


```python
if MAIN:
    print("splitting up 2.1 input by seqpos")
    tests.t_make_ex3_corr(make_ex3_corr)
    print("\nEx3a: first with real open proportion")
    ex3a, loss3a = paren_experiment(ex3_circuit, ds, make_ex3_corr(adjusted=False))
    check_loss(loss3a, 1.124, 0.1)
    print("\nEx3b: now with adjusted open proportion")
    ex3b, loss3b = paren_experiment(ex3_circuit, ds, make_ex3_corr(adjusted=True))
    check_loss(loss3b, 1.14, 0.1)
    if PRINT_CIRCUITS:
        ex3b.print()

```

## Experiment 4: All together
Now, combine experiments 2 (phi rewrite) and 3 (with adj. proportion)! Expected loss is ~1.64


```python
ex4_circuit = circuit
ex4_corr = Correspondence()
if MAIN and PRINT_CIRCUITS:
    printer.print(ex4_circuit)
if MAIN:
    print("\nEx4: Ex2b (1.0 and 2.0 phi rewrite) + Ex3b (2.1 split by seqpos with p_adj)")
    ex4, loss4 = paren_experiment(ex4_circuit, ds, ex4_corr)
    check_loss(loss4, 1.7, 0.1)

```