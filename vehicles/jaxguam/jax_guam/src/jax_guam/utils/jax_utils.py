from typing import TypeVar

import jax
import jax.tree_util as jtu
import numpy as np

_PyTree = TypeVar("_PyTree")


def jax_use_double():
    jax.config.update("jax_enable_x64", True)


def get_cpu_device(idx: int = 0):
    return jax.devices("cpu")[idx]


def jax_use_cpu() -> None:
    ctx = jax.default_device(get_cpu_device())
    ctx.__enter__()


def jax2np(tree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, tree)


def tree_mac(accum: _PyTree, scalar: float, other: _PyTree, strict: bool = True) -> _PyTree:
    """Tree multiply and accumulate. Return accum + scalar * other, but for pytree."""

    def mac_inner(a, o):
        if strict:
            assert a.shape == o.shape
        return a + scalar * o

    return jtu.tree_map(mac_inner, accum, other)


def tree_add(t1: _PyTree, t2: _PyTree):
    return jtu.tree_map(lambda a, b: a + b, t1, t2)


def tree_inner_product(coefs: list[float], trees: list[_PyTree]) -> _PyTree:
    def tree_inner_product_(*arrs_):
        arrs_ = list(arrs_)
        out = sum([c * a for c, a in zip(coefs, arrs_)])
        return out

    assert len(coefs) == len(trees)
    return jtu.tree_map(tree_inner_product_, *trees)
