import pandas as pd
import os


class ContractManager:
    def __init__(self, config_path=None):
        """
        Loads the contract specifications and prepares them for lookup.
        """
        if config_path is None:
            # We calculate the absolute path to the project root relative to this file
            # This file is in: project_root/python_modules/utilities/contract_manager.py
            # So we go up 3 levels to reach the project root.
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "data", "config", "contract_specs.csv")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Contract specs not found at {config_path}")

        # We load the data and set 'Instrument' as the index for lightning-fast lookups
        self.specs = pd.read_csv(config_path)
        self.specs.set_index('instrument', inplace=True)

    def get_contract_info(self, instrument_name):
        """
        Returns all specifications for a given instrument as a dictionary.
        """
        if instrument_name in self.specs.index:
            return self.specs.loc[instrument_name].to_dict()
        else:
            print(f"Warning: instrument '{instrument_name}' not found in specs.")
            return None

    def get_point_value(self, instrument_name):
        """
        Shortcut to get just the dollar value of a 1-point move.
        """
        info = self.get_contract_info(instrument_name)
        return info.get('PointValue', 0) if info else 0


# Beginner Tip: We create one instance here so it's ready to use
# This is often called a 'Singleton' pattern
manager = ContractManager()