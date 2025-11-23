import unittest
import numpy as np
import pandas as pd
from src.features.engineer import calculate_features, create_sequences
from src.ga.optimizer import WheelOptimizer

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=20),
            'id': [f'{i:05d}' for i in range(20)],
            'num1': np.random.randint(1, 35, 20),
            'num2': np.random.randint(1, 35, 20),
            'num3': np.random.randint(1, 35, 20),
            'num4': np.random.randint(1, 35, 20),
            'num5': np.random.randint(1, 35, 20),
            'special': np.random.randint(1, 12, 20)
        })

    def test_feature_calculation(self):
        df_feat = calculate_features(self.df)
        self.assertIn('sum_main', df_feat.columns)
        self.assertIn('even_count', df_feat.columns)
        self.assertIn('decade_1', df_feat.columns)

    def test_sequence_creation(self):
        df_feat = calculate_features(self.df)
        X, y1, y2 = create_sequences(df_feat, lookback=5)
        self.assertEqual(X.shape[0], 20-5)
        self.assertEqual(X.shape[1], 5) # lookback
        # 35 main + 8 basic + 35 hotness + 35 gap = 113
        self.assertEqual(X.shape[2], 113)

class TestGA(unittest.TestCase):
    def test_optimization(self):
        probs = np.random.rand(35)
        probs = probs / probs.sum()
        ga = WheelOptimizer(probs, None, ticket_budget=5, generations=2)
        best, score = ga.optimize()
        self.assertEqual(len(best), 5)
        self.assertTrue(score > 0)

if __name__ == '__main__':
    unittest.main()
