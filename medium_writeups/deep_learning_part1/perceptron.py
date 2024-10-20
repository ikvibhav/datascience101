import tensorflow as tf

# Create a simple perceptron
def create_perceptron(num_units: int = 1, input_size: int = 1) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units = num_units, input_shape=[input_size])
    ])


if __name__ == "__main__":
    # Simple Perceptron with 3 input features
    print("------------Simple Perceptron-----------------------")
    print(create_perceptron(num_units=1, input_size=3).count_params())

    print("------------Varying Input Shape-----------------------")
    # Simple Perceptron with varying number of inputs
    trainable_params = {}
    for i in range(1, 5):
        model = create_perceptron(num_units=1, input_size=i)
        trainable_params[i] = model.count_params()

    for key, val in trainable_params.items():
        print(f"Input shape: {key} -> Trainable parameters: {val}")

    print("------------Varying Number of Units and Input Shape-----------------------")
    # Create perceptrons with varying number of units and inputs
    trainable_params = {}
    for i in range(1, 5):
        for j in range(1, 5):
            model = create_perceptron(num_units=i, input_size=j)
            trainable_params[(i, j)] = model.count_params()
    
    for key, val in trainable_params.items():
        print(f"Number of units: {key[0]}, Input shape: {key[1]} -> Trainable parameters: {val} ({key[0]} x {key[1]} + {key[0]})")