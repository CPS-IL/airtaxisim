from typing import TypeVar

import numpy as np
import scipy
from typing_extensions import Self

TState = TypeVar("TState")
TFloat = TypeVar("TFloat")


class BatchSpline:
    def __init__(self, b_spl: list[scipy.interpolate.UnivariateSpline]):
        self.b_spl = b_spl

    def __call__(self, T_x: TFloat) -> TState:
        return np.stack([spl(T_x) for spl in self.b_spl], axis=-1)

    def derivative(self, n: int = 1) -> Self:
        return BatchSpline([spl.derivative(n) for spl in self.b_spl])


def get_spline(T_x: TFloat, T_y: TState, w: TFloat = None, k: int = 3, s: float = None):
    assert len(T_x) == len(T_y)
    args = dict(w=w, k=k, s=s)

    if T_y.ndim == 1:
        return scipy.interpolate.UnivariateSpline(T_x, T_y, **args)

    assert T_y.ndim == 2
    nx = T_y.shape[1]
    b_spl = [scipy.interpolate.UnivariateSpline(T_x, T_y[:, ii], **args) for ii in range(nx)]

    return BatchSpline(b_spl)
