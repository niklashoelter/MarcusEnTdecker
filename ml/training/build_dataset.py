import pandas as pd
from pathlib import Path
import ast
from typing import Dict, Tuple, Optional


class DataLoader:
    def __init__(self,
        datapath: Path
    ):
        '''
        Read the dataset CSV at `datapath`.
        '''
        self.input_data = pd.read_csv(datapath).copy()
        self.column_names = self.input_data.columns.tolist()
        self.target_columns = None

    def compute_targets(self,
        gibbs_singlet_column: str = 'Gibbs_S',
        gibbs_triplet_column: str = 'Gibbs_T',
        energy_singlet_singlet_column: str = 'E_S_geometry_S_energy',
        energy_triplet_triplet_column: str = 'E_T_geometry_T_energy',
        energy_triplet_singlet_column: str = 'E_T_geometry_S_energy',
        energy_singlet_triplet_column: str = 'E_S_geometry_T_energy',
        target_columns: list = ['adiabatic_energy', 'lambda_acceptor_singlet', 'lambda_donor_triplet'],
        return_compact: bool = True,
        Eh_to_kcal: float = 627.50962,
        filter_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
    ) -> pd.DataFrame:
        '''
        Compute the target columns (kcal/mol) from the raw energies (Hartree).

        :param Eh_to_kcal: Hartree to kcal/mol.
        :param gibbs_singlet_column, gibbs_triplet_column: Gibbs corrections of S0 and T1.
        :param energy_X_Y_column: energy of state Y at the geometry of state X.
        :param target_columns: any of adiabatic_energy, lambda_acceptor_singlet,
            lambda_donor_triplet, vertical_excitation, vertical_emission.
        :param filter_limits: {column: (min, max)}, None = open side. None: adiabatic_energy
            in [15, 120] kcal/mol, lambdas >= 0. {} disables filtering.
        :return: DataFrame with the target columns.
        '''

        self.target_columns = target_columns
        # Extract numpy arrays
        gibbs_s = self.input_data[gibbs_singlet_column].to_numpy()
        gibbs_t = self.input_data[gibbs_triplet_column].to_numpy()
        es = self.input_data[energy_singlet_singlet_column].to_numpy()
        et = self.input_data[energy_triplet_triplet_column].to_numpy()

        # Compute target columns
        for col in target_columns:
            if col == 'spin_population' and col in self.input_data.columns:
                self.input_data[col] = self.input_data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                continue

            if col == 'adiabatic_energy':
                # E_T = (Gibbs_T + E_T_geometry_T) - (Gibbs_S + E_S_geometry_S)
                self.input_data['adiabatic_energy'] = ((gibbs_t + et) - (gibbs_s + es)) * Eh_to_kcal

            elif col == 'lambda_acceptor_singlet':
                # lambda_S0_T1 = E_T_geometry_S - E_S_geometry_S (S0 surface)
                self.input_data['lambda_acceptor_singlet'] = (self.input_data[energy_triplet_singlet_column] - self.input_data[energy_singlet_singlet_column]) * Eh_to_kcal

            elif col == 'lambda_donor_triplet':
                # lambda_T1_S0 = E_S_geometry_T - E_T_geometry_T (T1 surface)
                self.input_data['lambda_donor_triplet'] = (self.input_data[energy_singlet_triplet_column] - self.input_data[energy_triplet_triplet_column]) * Eh_to_kcal

            elif col == 'vertical_excitation':
                # vertical S0 -> T1 excitation = E_S_geometry_T - E_S_geometry_S
                self.input_data['vertical_excitation'] = (self.input_data[energy_singlet_triplet_column] - self.input_data[energy_singlet_singlet_column]) * Eh_to_kcal

            elif col == 'vertical_emission':
                # vertical T1 -> S0 emission = E_T_geometry_T - E_T_geometry_S
                self.input_data['vertical_emission'] = (self.input_data[energy_triplet_triplet_column] - self.input_data[energy_triplet_singlet_column]) * Eh_to_kcal

            else:
                raise ValueError(f"Unknown target column: {col}")

        # Apply default filters if filter_limits is None, or custom filters if provided
        if filter_limits is None:
            # Default limits
            default_limits = {
                'spin_population': (None, None),
                'adiabatic_energy': (15, 120),
                'lambda_acceptor_singlet': (0, None),
                'lambda_donor_triplet': (0, None)
            }
            # Only apply limits for columns that exist in target_columns
            applicable_limits = {k: v for k, v in default_limits.items() if k in target_columns}
            if applicable_limits:
                self.filter_by_limits(limits=applicable_limits, inplace=True)
        elif filter_limits:  # Non-empty dict provided
            self.filter_by_limits(limits=filter_limits, inplace=True)
        # If filter_limits is an empty dict {}, skip filtering entirely

        if return_compact:
            return self.return_compact_data()
        return self.input_data

    def filter_by_limits(self,
        limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        inplace: bool = True
    ) -> pd.DataFrame:
        '''
        Drop rows outside the limits.

        :param limits: {column: (min, max)}, None = open side; for spin_population every
            atom must lie within.
        :param inplace: modify self.input_data, else return a filtered copy.
        :return: Filtered DataFrame.
        '''
        if limits is None:
            limits = {}

        if inplace:
            df = self.input_data
        else:
            df = self.input_data.copy()

        initial_count = len(df)

        for column, (min_val, max_val) in limits.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in the dataset. Available columns: {df.columns.tolist()}")

            # Handle spin_population column specially (it's a list)
            if column == 'spin_population':
                if min_val is not None:
                    # For spin_population, check if any value in the list is below min_val
                    mask = df[column].apply(lambda x: all(v >= min_val for v in x) if isinstance(x, list) else x >= min_val)
                    df = df[mask]
                if max_val is not None:
                    mask = df[column].apply(lambda x: all(v <= max_val for v in x) if isinstance(x, list) else x <= max_val)
                    df = df[mask]
            else:
                if min_val is not None:
                    df = df[df[column] >= min_val]
                if max_val is not None:
                    df = df[df[column] <= max_val]

        filtered_count = len(df)

        # Format limits for display
        limits_str = ", ".join([
            f"{col}: [{min_val if min_val is not None else '-∞'}, {max_val if max_val is not None else '∞'}]"
            for col, (min_val, max_val) in limits.items()
        ])
        print(f"Filtered dataset: {initial_count} -> {filtered_count} rows ({initial_count - filtered_count} rows removed)")
        if limits:
            print(f"Applied limits: {limits_str}")

        if inplace:
            self.input_data = df

        return df

    def show_columns(self):
        print("Available columns in the dataset:")
        for col in self.column_names:
            print(f"- {col}")

    def return_entire_df(self):
        return self.input_data

    def return_compact_data(self,
                            idx_column:str = 'idx',
                            smiles_column: str = 'smiles'
    ) -> pd.DataFrame:
        relevant_columns = [idx_column, smiles_column] + [col for col in self.target_columns if col not in [smiles_column]]
        return self.input_data[relevant_columns]