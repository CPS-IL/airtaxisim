from jax_guam.guam_types import Power, PwrCmd


def power_system(pwrCmd: PwrCmd) -> Power:
    return Power(CtrlSurfacePwr=pwrCmd.CtrlSurfacePwr, EnginePwr=pwrCmd.EnginePwr)
