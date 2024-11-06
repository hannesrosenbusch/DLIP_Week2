import numpy as np
from tasks import step

# Test cases for Task 1: step function
def test_step_positive():
    assert step(5) == 1, "Failed on positive input"

def test_step_negative():
    assert step(-3) == -1, "Failed on negative input"

def test_step_zero():
    assert step(0) == -1, "Failed on zero input"
