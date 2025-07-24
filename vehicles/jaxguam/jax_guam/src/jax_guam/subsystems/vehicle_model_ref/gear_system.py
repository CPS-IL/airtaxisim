from jax_guam.utils.jax_types import FloatScalar


def gear_system(gear_cmd: FloatScalar) -> FloatScalar:
    gain = 1.0
    gear_out = gain * gear_cmd
    return gear_out
