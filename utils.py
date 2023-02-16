import tempfile
import os
import time
import torch as t
from torch import nn
import transformers
import joblib
import requests
import logging
import http
from functools import wraps
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from typing import Optional, Iterator, cast, TypeVar, Generic, Callable

mem = joblib.Memory(tempfile.gettempdir() + "/joblib_cache")
DEBUG_TOLERANCES = os.getenv("DEBUG_TOLERANCES")


@mem.cache
def load_pretrained_gpt() -> GPT2LMHeadModel:
    """Load the HuggingFace GPT-2.

    On first use this downloads about 500MB from the Internet.
    Later uses should hit the cache and take under 1s to load.
    """
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2")


@mem.cache
def load_pretrained_bert() -> BertForMaskedLM:
    """Load the HuggingFace BERT.

    Supresses the spurious warning about some weights not being used.
    """
    logger = logging.getLogger("transformers.modeling_utils")
    was_disabled = logger.disabled
    logger.disabled = True
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    logger.disabled = was_disabled
    return cast(BertForMaskedLM, bert)


def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    """Assert that actual and expected are exactly equal (to floating point precision)."""
    mask = actual == expected
    if not mask.all().item():
        bad = mask.nonzero()
        msg = f"Did not match at {len(bad)} indexes: {bad[:10]}{'...' if len(bad) > 10 else ''}"
        raise AssertionError(f"{msg}\nActual:\n{actual}\nExpected:\n{expected}")


def test_is_equal(actual: t.Tensor, expected: t.Tensor, test_name: str) -> None:
    try:
        run_and_report(assert_all_equal, test_name, actual, expected)
    except AssertionError as e:
        print(f"Test failed: {test_name}")
        raise e


def assert_shape_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    right = rtol * expected.abs()
    num_wrong = (left > right).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")
    elif DEBUG_TOLERANCES:
        print(f"Test passed with max absolute deviation of {left.max()}")


def allclose_atol(actual: t.Tensor, expected: t.Tensor, atol: float) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    num_wrong = (left > atol).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")
    elif DEBUG_TOLERANCES:
        print(f"Test passed with max absolute deviation of {left.max()}")


def allclose_scalar(actual: float, expected: float, rtol=1e-4) -> None:
    left = abs(actual - expected)
    right = rtol * abs(expected)
    wrong = left > right
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")
    elif DEBUG_TOLERANCES:
        print(f"Test passed with absolute deviation of {left}")


def allclose_scalar_atol(actual: float, expected: float, atol: float) -> None:
    left = abs(actual - expected)
    wrong = left > atol
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")
    elif DEBUG_TOLERANCES:
        print(f"Test passed with absolute deviation of {left}")


def report_success(testname):
    """POST to the server indicating success at the given test.

    Used to help the TAs know how long each section takes to complete.
    """
    server = os.environ.get("MLAB_SERVER")
    email = os.environ.get("MLAB_EMAIL")
    if server:
        if email:
            r = requests.post(
                server + "/api/report_success",
                json=dict(email=email, testname=testname),
            )
            if r.status_code != http.HTTPStatus.NO_CONTENT:
                raise ValueError(f"Got status code from server: {r.status_code}")
        else:
            raise ValueError(f"Server set to {server} but no MLAB_EMAIL set!")
    else:
        if email:
            raise ValueError(f"Email set to {email} but no MLAB_SERVER set!")
        else:
            return  # local dev, do nothing


# Map from qualified name "test_w2d3.test_unidirectional_attn" to whether this test was passed in the current interpreter session
# Note this can get clobbered during autoreload
TEST_FN_PASSED = {}


def report(test_func):
    name = f"{test_func.__module__}.{test_func.__name__}"
    # This can happen when using autoreload, so don't complain about it.
    # if name in TEST_FN_PASSED:
    #     raise KeyError(f"Already registered: {name}")
    TEST_FN_PASSED[name] = False

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        return run_and_report(test_func, name, *args, **kwargs)

    return wrapper


def run_and_report(test_func: Callable, name: str, *test_func_args, **test_func_kwargs):
    start = time.time()
    out = test_func(*test_func_args, **test_func_kwargs)
    elapsed = time.time() - start
    print(f"{name} passed in {elapsed:.2f}s.")
    if not TEST_FN_PASSED.get(name):
        report_success(name)
        TEST_FN_PASSED[name] = True
    return out


def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


from torch.nn.modules.module import _addindent

T = TypeVar("T")


class StaticModuleList(nn.ModuleList, Generic[T]):
    """ModuleList where the user vouches that it only contains objects of type T.

    This allows the static checker to work instead of only knowing that the contents are Modules.
    """

    # TBD lowpri: is it possible to do this just with signatures, without actually overriding the method bodies to add a cast?

    def __getitem__(self, index: int) -> T:
        return cast(T, super().__getitem__(index))

    def __iter__(self) -> Iterator[T]:
        return cast(Iterator[T], iter(self._modules.values()))

    def __repr__(self):
        # CM: modified from t.nn.Module.__repr__
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        modules = iter(self._modules.items())
        key, module = next(modules)
        n_rest = sum(1 for _ in modules)
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines + [f"+ {n_rest} more..."]

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


if __name__ == "__main__":
    ml = StaticModuleList([nn.Linear(1, 2) for _ in range(5)])
    print(ml)
