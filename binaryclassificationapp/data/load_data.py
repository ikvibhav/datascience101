from ucimlrepo import fetch_ucirepo

def load_ucimlrepo_data(dataset_id: int, class_names: list) -> tuple:
    dataset = fetch_ucirepo(id=dataset_id)
    return dataset.data.features, dataset.data.targets, class_names

def load_ucimlrepo_mushroomdata() -> tuple:
    return load_ucimlrepo_data(73, ["edible", "poisonous"])

def load_ucimlrepo_breastcancer() -> tuple:
    return load_ucimlrepo_data(15, ["benign", "malignant"])

def load_ucimlrepo_diabetes_data() -> tuple:
    return load_ucimlrepo_data(891, ["positive", "negative"])