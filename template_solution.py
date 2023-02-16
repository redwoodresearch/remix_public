# %%
"""

# Blank Template Solution File

You can use this file as a template to write solution notebooks that easily convert into exercises. In a real solution notebook, this blurb before the table of contents would give an introduction to the topic, say what the final payoff of the day will be, and get them hyped up to continue through the material.

<!-- toc -->

The above table of contents is automatically generated based on the headings. Use two hash marks `##` for most topics, and three hash marks `###` for subsections within a topic.

## Readings

- [The Zen of Python](https://peps.python.org/pep-0020/)


### How do I build the corresponding instructions file?

- One way is to run `python build_instructions.py template_solution.py`.
- If you run `python build_all.py`, this will keep running in the background and automatically rebuild each solution file on file change.
- You can set up a pre-commit hook so that it automatically rebuilds changed files.
"""
# %%

# Below is a common preamble you can copy and paste
# Note that regular comments like this do NOT appear in the instructions,
# meaning they're suitable for messages to other staff but NOT useful for
# explaining why code is a certain way - use docstrings for that.
# Keep in mind that participants will have access to the full solution file and
# can peek at their own discretion.
import os
import sys
import torch as t
from torch import nn

MAIN = __name__ == "__main__"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

# This environment variable is set on GitHub when continuous integration is running.
# It's good to run as much of the file as possible during CI to catch regressions,
# but in many cases this isn't possible because CI doesn't have access to model weights.
# It's fine to just exit and skip everything if it isn't practical to run the exercise in CI.
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)


# %%
"""
## Solution blocks

This is an LaTeX equation: $ c = a + b $

When we refer to Python names, backticks can be used to render the name in a monospace font. For example: `c` depends on `a` and `b`.

This is a graph:

```mermaid

graph TD
    a --> c 
    b --> c
```

Pretend we ask the participants to implement c = a + b.
"""
# %%
a = 5
b = -2

if "SOLUTION":
    c = a + b
else:
    """Define C here"""

# %%
"""
### Solution blocks in functions/methods

"""

# %%
class ExampleRelu(nn.Module):
    def __init__(self):
        """This docstring is part of the instruction file."""
        super().__init__()  # This is also part of the instruction file
        "SOLUTION"
        # Everything below the "SOLUTION" line is not in the instruction file.
        print("In the constructor")


def regular_function(a, b):
    "SOLUTION"
    # The below is not in the instruction file.
    return a + b


# %%
"""
## Spoilers / Hints

Spoilers can be hidden by default using `<details>` and `<summary>` tags.

Use these if you present text exercises that they should solve on their own before peeking at the spoiler, or to preempt questions. For example, you can provide common error messages as a spoiler and they can click to see the resolution of the error.

Another use of spoilers is to provide hints that don't completely solve the problem, so they can just look at the spoiler instead of calling a TA or looking at the solution file.

<details>

<summary>Solution - participants can click this to see the contents </summary>

This is the inside of the solution block.

You can still use equations like $ y = ax $ and Markdown like **bold** in here.

</details>

"""

# %%
