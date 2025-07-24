from typing import Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int

Arr = Union[np.ndarray, Array]

AnyFloat = Float[Arr, "*"]
FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

Vec1 = Float[Arr, "1"]
Vec2 = Float[Arr, "2"]
Vec3 = Float[Arr, "3"]
Vec4 = Float[Arr, "4"]
Vec5 = Float[Arr, "5"]
Vec6 = Float[Arr, "6"]
Vec8 = Float[Arr, "8"]
Vec10 = Float[Arr, "10"]
Vec11 = Float[Arr, "11"]

Vec1_1 = Float[Arr, "1 1"]
Vec2_1 = Float[Arr, "2 1"]
Vec3_1 = Float[Arr, "3 1"]
Vec4_1 = Float[Arr, "4 1"]
Vec5_1 = Float[Arr, "5 1"]
Vec6_1 = Float[Arr, "6 1"]
Vec9_1 = Float[Arr, "9 1"]
Vec12_1 = Float[Arr, "12 1"]
Vec13_1 = Float[Arr, "13 1"]
Vec25_1 = Float[Arr, "25 1"]

RowVec4 = Float[Arr, "1 4"]
RowVec6 = Float[Arr, "1 6"]
RowVec8 = Float[Arr, "1 8"]

FMVec = Vec6
FMVec_1 = Vec6_1
PropVec = Float[Arr, "n_props"]

Quat = Vec4
Quat_1 = Vec4_1

Vec12 = Float[Arr, "12"]
Vec13 = Float[Arr, "13"]
Vec14 = Float[Arr, "14"]
Vec15 = Float[Arr, "15"]


Vec9 = Float[Arr, "1 9"]  # TODO: 1x9?

Mat1 = Float[Arr, "1 1"]
Mat3 = Float[Arr, "3 3"]
Mat4 = Float[Arr, "4 4"]
Mat4_11 = Float[Arr, "4 11"]

Mat11 = Float[Arr, "11 11"]
Mat12 = Float[Arr, "12 12"]
Mat3_2 = Float[Arr, "3 2"]
Mat3_4 = Float[Arr, "3 4"]
Mat1_4 = Float[Arr, "1 4"]


Mat31 = Float[Arr, "3 1"]

Mat3_11 = Float[Arr, "3 11"]
Mat3_12 = Float[Arr, "3 12"]
Mat6_14 = Float[Arr, "6 14"]
Mat14 = Float[Arr, "14 14"]

State = Float[Arr, "nx"]

AfVec = Float[Arr, "n_strips"]
AfBool = Bool[Arr, "n_strips"]

Af1 = Float[Arr, "n_strips 1"]
Af2 = Float[Arr, "n_strips 2"]
Af3 = Float[Arr, "n_strips 3"]
Af4 = Float[Arr, "n_strips 4"]
Af6 = Float[Arr, "n_strips 6"]
Af8 = Float[Arr, "n_strips 8"]

Force = Vec3
Moment = Vec3
