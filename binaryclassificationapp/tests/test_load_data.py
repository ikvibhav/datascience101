import pytest
from data.load_data import (
    load_ucimlrepo_mushroomdata,
    load_ucimlrepo_breastcancer,
    load_ucimlrepo_diabetes_data,
)

def test_load_ucimlrepo_mushroomdata():
    X, Y, class_names = load_ucimlrepo_mushroomdata()
    assert X is not None
    assert Y is not None
    assert class_names == ["edible", "poisonous"]

def test_load_ucimlrepo_breastcancer():
    X, Y, class_names = load_ucimlrepo_breastcancer()
    assert X is not None
    assert Y is not None
    assert class_names == ["benign", "malignant"]

# Takes a long time to run due to 200K rows
# def test_load_ucimlrepo_diabetes_data():
#     X, Y, class_names = load_ucimlrepo_diabetes_data()
#     assert X is not None
#     assert Y is not None
#     assert class_names == ["positive", "negative"]