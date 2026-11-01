import math

def eyring_equation(dG_dagger, T=298.15, units="kcal/mol"):
    """
    Compute rate constant k (s^-1) from activation free energy using Eyring equation.

    Parameters:
        dG_dagger (float): Activation free energy
        T (float): Temperature in Kelvin (default 298.15 K)
        units (str): "kJ/mol", "J/mol", "kcal/mol", or "eV"

    Returns:
        k (float): Rate constant in s^-1
    """

    # Constants (SI units)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    h = 6.62607015e-34      # Planck constant (J*s)
    R = 8.314462618         # Gas constant (J/mol/K)

    # Convert dG‡ to J/mol
    if units == "kJ/mol":
        dG_dagger *= 1000
    elif units == "kcal/mol":
        dG_dagger *= 4184
    elif units == "eV":
        dG_dagger *= 96485.33212  # eV → J/mol
    elif units == "J/mol":
        pass
    else:
        raise ValueError("units must be 'kJ/mol', 'kcal/mol', 'J/mol', or 'eV'")

    # Eyring equation
    k = (k_B * T / h) * math.exp(-dG_dagger / (R * T))

    return k