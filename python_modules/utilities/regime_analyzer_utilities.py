import pandas as pd
import numpy as np
from python_modules.utilities.regimeclassifierkmeans import classifier

class RegimeAnalyzer:
    """
    Translates abstract ML clusters into actionable trading environments.
    """
    
    @staticmethod
    def get_regime_profiles(historical_data):
        """
        Analyzes the fitted classifier to describe what each regime ID actually means.
        """
        # Get the feature data with cluster assignments
        df_features = classifier.fit(historical_data)
        
        # Calculate averages for each cluster to identify their 'personality'
        profiles = df_features.groupby('cluster').mean()
        
        # Add labels based on simple heuristics
        def label_regime(row):
            if row['volatility'] > df_features['volatility'].quantile(0.7):
                return "High Vol / Crisis"
            if row['trend_dist'] > 0 and row['momentum'] > 0:
                return "Bullish Trend"
            if row['trend_dist'] < 0 and row['momentum'] < 0:
                return "Bearish Trend"
            return "Quiet / Mean Reverting"

        profiles['description'] = profiles.apply(label_regime, axis=1)
        return profiles

    @staticmethod
    def check_strategy_fit(current_data, best_regime_id, ml_confidence):
        """
        Compares live market state to a strategy's preferred historical regime.
        """
        current_regime = classifier.predict_regime(current_data)
        
        is_match = (current_regime == best_regime_id)
        
        # Basic scoring: 1.0 if match, 0.5 if mismatch but high confidence, 0.2 if mismatch
        fit_score = 1.0 if is_match else (0.5 if ml_confidence > 0.70 else 0.2)
        
        return {
            "current_regime": current_regime,
            "target_regime": best_regime_id,
            "is_ideal_fit": is_match,
            "sizing_multiplier": fit_score
        }

# Singleton for project-wide use
regime_pm = RegimeAnalyzer()
