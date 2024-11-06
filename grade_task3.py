import numpy as np
from tasks import step, ReLu, neural_net_layer

def test_neural_net_layer_basic():
    inputs = np.array([[1, 2], [3, 4]])
    weights = np.array([1, -1])
    expected_output = np.array([0, 0])  # (1*1 + 2*(-1)) = -1 -> 0 (after ReLu), (3*1 + 4*(-1)) = -1 -> 0 (after ReLu)
    np.testing.assert_array_equal(neural_net_layer(inputs, weights), expected_output, "Failed on basic 2D array")

def test_neural_net_layer_custom():
    inputs = np.array([[2, 3], [1, 1]])
    weights = np.array([1, 2])
    expected_output = np.array([8, 3])  # (2*1 + 3*2) = 8, (1*1 + 1*2) = 3
    np.testing.assert_array_equal(neural_net_layer(inputs, weights), expected_output, "Failed on different inputs")

def test_neural_net_layer_with_negatives():
    inputs = np.array([[1, -1], [-2, 3]])
    weights = np.array([-2, 1])
    expected_output = np.array([0, 7])  # ReLu([-3, 7]) = [0, 7]
    np.testing.assert_array_equal(neural_net_layer(inputs, weights), expected_output, "Failed on negative matrix values")