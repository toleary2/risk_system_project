import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RegimeClassifier:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, data, window=20):
        """
        Creates features based on the last 'window' days:
        - Volatility (Standard Deviation of returns)
        - Trend (Distance from Moving Average)
        - Momentum (RSI or Rate of Change)
        """
        df = data.copy()
        returns = df['close'].pct_change()
        
        features = pd.DataFrame(index=df.index)
        features['volatility'] = returns.rolling(window).std()
        features['trend_dist'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).mean()
        features['momentum'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        return features.dropna()

    def fit(self, data):
        """Trains the K-Means model on historical data."""
        features = self.prepare_features(data)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        self.is_fitted = True
        
        # Assign names to clusters based on volatility/trend (Human readable)
        clusters = self.model.predict(scaled_features)
        features['cluster'] = clusters
        return features

    def predict_regime(self, recent_data):
        """Predicts the regime for a specific point in time."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted on historical data first.")
        
        # Prepare and scale the most recent features
        feat = self.prepare_features(recent_data).tail(1)
        if feat.empty: return -1
        
        scaled_feat = self.scaler.transform(feat)
        return self.model.predict(scaled_feat)[0]

# Singleton instance
classifier = RegimeClassifier()
