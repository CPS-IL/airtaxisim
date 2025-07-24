import pdb

import ipdb
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from loguru import logger

from jax_guam.guam_types import EulerAngles
from jax_guam.utils.jax_types import Arr, FloatScalar, IntScalar, Mat3, Quat_1, Vec2_1, Vec3, Vec3_1


def quaternion_vector_transformation(q: Quat_1, v: Vec3_1 | Vec3) -> Vec3:
    """Rotate a vector by a quaternion."""
    assert q.shape == (4, 1)
    assert v.shape == (3, 1) or v.shape == (3,)
    q_0 = q[0]
    q = q[1:, :]
    q = q.reshape((3,))
    v = v.reshape((3,))
    plus_1 = q * (2 * jnp.dot(q, v))
    plus_2 = (q_0 * q_0 - jnp.dot(q, q)) * v
    minus_3 = 2 * q_0 * jnp.cross(q, v, axis=0)
    V_b = plus_1 + plus_2 - minus_3
    assert V_b.shape == (3,)
    return V_b


def quaternion_vector_transformation2(q_m, v_m):
    V_b = []
    for i in range(q_m.shape[1]):
        q = q_m[:, i]
        v = v_m[:, i]
        q_0 = q[0]
        q = q[1:]
        q = q.reshape((3,))
        v = v.reshape((3,))
        plus_1 = q * (2 * jnp.dot(q, v))
        plus_2 = (q_0 * q_0 - jnp.dot(q, q)) * v
        minus_3 = 2 * q_0 * jnp.cross(q, v, axis=0)
        V_b.append(plus_1 + plus_2 - minus_3)
    return jnp.array(V_b)


def quaternion_inverse(q: Quat_1) -> Quat_1:
    assert q.shape == (4, 1)
    inv_q = quaternion_conjugate(q) / jnp.sum(q**2)
    return inv_q


def quaternion_multiplication(p, q):
    first = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    second = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
    third = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    forth = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]

    return jnp.array([first, second, third, forth])


def quaternion_conjugate(q: Quat_1) -> Quat_1:
    assert q.shape == (4, 1)
    CONSTANT = np.array([[1], [-1], [-1], [-1]])
    q_star = CONSTANT * q
    assert q_star.shape == (4, 1)
    return q_star


def quaternion_rotz(angle: FloatScalar) -> Quat_1:
    quat = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])[:, None]
    assert quat.shape == (4, 1)
    return quat


def single_axis_quaternion(r, axis):
    if axis == 0:
        mul = jnp.array([[1], [0], [0]])
    elif axis == 1:
        mul = jnp.array([[0], [1], [0]])
    else:
        mul = jnp.array([[0], [0], [1]])
    return jnp.concatenate([jnp.cos(r * 0.5).reshape((1, 1)), jnp.sin(r * 0.5) * mul])


def quaternion_to_DCM(q):
    e_1 = q[0].reshape(1)
    e_2t4 = q[1:]
    sum_1 = (e_1**2 - jnp.sum(e_2t4**2)) * jnp.eye(3)
    sum_2 = jnp.outer(e_2t4, e_2t4) * 2
    minus_3 = cross_product_matrix(e_2t4) * e_1 * 2
    return (sum_1 + sum_2 - minus_3).T


def cross_product_matrix(r):
    mat = jnp.array([r[0], r[2], r[1], r[2], r[1], r[0], r[1], r[0], r[2]]).reshape(9)
    CON = jnp.array([0, 1, -1, -1, 0, 1, 1, -1, 0])
    return (CON * mat).reshape((3, 3))


def nquaternion_to_Euler(Q_i2b: Quat_1) -> EulerAngles:
    assert Q_i2b.shape == (4, 1)
    # Q_b2a = nquaternion_conjugate(Q_i2b)
    Q_b2a = quaternion_conjugate(Q_i2b)

    U1_3 = np.array([[1], [0], [0]])
    U1_2 = np.array([[0], [1], [0]])
    U1_1 = np.array([[0], [0], [1]])
    U1_irt3_0 = quaternion_vector_transformation(Q_b2a, jnp.dot(U1_3, Q_b2a[0] + 1)).reshape((3, 1))
    U1_irt2_0 = quaternion_vector_transformation(Q_b2a, jnp.dot(U1_2, Q_b2a[0] + 1)).reshape((3, 1))

    mat1: Vec2_1 = jnp.dot(jnp.eye(2), jnp.flip(U1_irt3_0, 0)[1:])  # Note because 1==1 is true, omit the TF selector
    mat1_vec = mat1.squeeze(-1)
    ROT1: FloatScalar = jnp.arctan2(mat1_vec[0], mat1_vec[1])
    assert ROT1.shape == tuple()

    Q_0to1 = jnp.concatenate(
        [jnp.cos(ROT1 * 0.5).reshape((1, 1)), jnp.dot(jnp.sin(ROT1 * 0.5), U1_1)], axis=0
    )  # Note is ROT1 1 or 1x1?
    U1_irt3_1 = quaternion_vector_transformation(Q_0to1, U1_irt3_0).reshape((3, 1))
    mat2: Vec2_1 = jnp.dot(jnp.array([[0, -1], [1, 0]]), jnp.delete(U1_irt3_1, 1, 0))  # Note because 1==3 is false
    mat2_vec = mat2.squeeze(-1)
    ROT2: FloatScalar = jnp.arctan2(mat2_vec[0], mat2_vec[1])
    assert ROT1.shape == tuple()

    Q_1to2 = jnp.concatenate(
        (jnp.cos(ROT2 * 0.5).reshape((1, 1)), jnp.dot(jnp.sin(ROT2 * 0.5), U1_2)), axis=0
    )  # Note is ROT2 1 or 1x1?
    Q_0to2 = quaternion_multiplication(Q_0to1, Q_1to2)
    U1_irt2_1 = quaternion_vector_transformation(Q_0to2, U1_irt2_0).reshape((3, 1))
    mat3: Vec2_1 = jnp.dot(jnp.eye(2), jnp.flip(U1_irt2_1, 0)[:2])  # Note because 1==3 is false
    mat3_vec = mat3.squeeze(-1)
    ROT3: FloatScalar = jnp.arctan2(mat3_vec[0], mat3_vec[1])
    assert ROT3.shape == tuple()

    mat4: Vec3_1 = jnp.array([[ROT1], [ROT2], [ROT3]])
    assert mat4.shape == (3, 1)
    mat5: Mat3 = jnp.concatenate((U1_1, U1_2, U1_3), axis=1)
    assert mat5.shape == (3, 3)
    mul: Vec3 = jnp.dot(mat5, mat4).squeeze()
    assert mul.shape == (3,)
    return EulerAngles(mul[0], mul[1], mul[2])


def nquaternion_conjugate(q: Quat_1) -> Quat_1:
    assert q.shape == (4, 1)
    mat = np.eye(4) * np.array([1, -1, -1, -1])
    return jnp.dot(mat, q)


def lookup(value, input_values, datas):
    left = None
    right = None
    for i, input_value in enumerate(input_values):
        if value == input_value:
            return datas[i]
        if value < input_value:
            right = i
            left = i - 1
    if left is None:  # larger than largest Note: only used two values, linear
        return (value - input_values[-1]) / (input_values[-1] - input_values[-2]) * (datas[-1] - datas[-2]) + datas[-1]
    if left == -1:  # smaller than smallest
        return datas[0] - (input_values[0] - value) / (input_values[1] - input_values[0]) * (datas[1] - datas[0])
    else:
        return datas[left] + (value - input_values[left]) / (input_values[right] - input_values[left]) * (
            datas[right] - datas[left]
        )


def get_lininterp_idx(value: FloatScalar, breaks: Float[Arr, "n_pts"]):
    assert breaks[0] < breaks[-1]
    breaks_jnp = breaks
    if not isinstance(breaks, jnp.ndarray):
        breaks_jnp = jnp.array(breaks)

    k = jnp.digitize(value, breaks, right=False) - 1
    f = (value - breaks_jnp[k]) / (breaks_jnp[k + 1] - breaks_jnp[k])

    is_within = (value >= breaks[0]) & (value <= breaks[-1])
    k = jnp.clip(k, 0, len(breaks) - 1)
    f = jnp.where(is_within, f, 0.0)

    return k, f


def matrix_interpolation(
    k_1: IntScalar, f_1: FloatScalar, k_2: IntScalar, f_2: FloatScalar, table_datas: Float[Arr, "data_dim dim1 dim2"]
):
    if len(table_datas.shape) == 3:
        # (1, data_dim, dim1, dim2)
        table_datas = jnp.expand_dims(table_datas, axis=0)

    dim0, data_dim, dim1, dim2 = table_datas.shape

    if not isinstance(table_datas, jnp.ndarray):
        # So we can index it.
        table_datas = jnp.array(table_datas)

    k_1_p1 = jnp.minimum(k_1 + 1, table_datas.shape[2] - 1)
    # (1, data_dim, dim2)
    first_1 = table_datas[:, :, k_1]
    first_2 = table_datas[:, :, k_1_p1]
    # (1, data_dim, dim2)
    first = first_1 + (first_2 - first_1) * f_1
    assert first.shape == (dim0, data_dim, dim2)

    k_2_p1 = jnp.minimum(k_2 + 1, table_datas.shape[2] - 1)
    # (1, data_dim)
    second_1 = first[:, :, k_2]
    # (1, data_dim)
    second_2 = first[:, :, k_2_p1]
    # (1, data_dim)
    second = second_1 + (second_2 - second_1) * f_2
    assert second.shape == table_datas.shape[:2]

    return second


def matrix_interpolation_np(k_1, f_1, k_2, f_2, table_datas):
    if len(table_datas.shape) == 3:
        table_datas = jnp.expand_dims(table_datas, axis=0)
    first_1 = table_datas[:, :, k_1]
    if k_1 + 1 == table_datas.shape[-2]:
        first = table_datas[:, :, k_1]
    else:
        first_2 = table_datas[:, :, k_1 + 1]
        first = first_1 + (first_2 - first_1) * f_1

    second_1 = first[:, :, k_2]
    if k_2 + 1 == first.shape[-1]:
        second = first[:, :, k_2]
    else:
        second_2 = first[:, :, k_2 + 1]
        second = second_1 + (second_2 - second_1) * f_2
    return second


def pseudo_inverse(W: Float[Arr, "nu nu"], B: Float[Arr, "3 nu"]) -> Float[Arr, "nu 3"]:
    """Compute lstsq(W, B^T) * (B W^-1 B^T)^-1"""
    W_inv = jnp.linalg.inv(W)
    return (jnp.linalg.lstsq(W, B.T, rcond=None)[0]).dot(jnp.linalg.inv(B.dot(W_inv).dot(B.T)))


def pseudo_inverse_np(W: Float[Arr, "nu nu"], B: Float[Arr, "3 nu"]) -> Float[Arr, "nu 3"]:
    """Compute lstsq(W, B^T) * (B W^-1 B^T)^-1"""
    W = np.array(W)
    B = np.array(B)

    W_inv = np.linalg.inv(W)
    return (np.linalg.lstsq(W, B.T, rcond=None)[0]).dot(np.linalg.inv(B.dot(W_inv).dot(B.T)))
