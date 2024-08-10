import pandas as pd
from preprocessing.encode_labels import encode_labels

def test_encode_labels():
    data = pd.DataFrame({
        'feature1': ['A', 'B', 'A', 'C'],
        'feature2': ['X', 'Y', 'X', 'Z']
    })
    encoded_data = encode_labels(data.copy())
    assert encoded_data['feature1'].nunique() == 3
    assert encoded_data['feature2'].nunique() == 3
    assert all(encoded_data['feature1'].isin([0, 1, 2]))
    assert all(encoded_data['feature2'].isin([0, 1, 2]))