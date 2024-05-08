from unittest import TestCase
from training import clean_data
import pandas as pd

class TestTrain(TestCase):
    def test_clean_data(self):
        df = pd.DataFrame({'survived' : [1,0,1], 
            'pclass' : [1, 2, 3],
            'sex' : ['male', 'female', 'male'],
            'age' : [20, 30, 40]})
    
    
        result_df = clean_data(df)

        self.assertEqual(result_df.shape[0], 3)
        self.assertEqual(result_df['sex'][0], 0)

if __name__ == '__main__':
    TestTrain().test_clean_data()