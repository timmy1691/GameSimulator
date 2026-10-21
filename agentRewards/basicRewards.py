import math
import numpy as np

def basicExpectedRewards(lastReward, perceivedReward, memoryWeight = 1):
    r_1 = lastReward
    q_1 = perceivedReward
    return (1- memoryWeight) * perceivedReward + memoryWeight*lastReward


def expectationUpdatebasic(currentPerception, currentReward, learning_rate):
    newReward = currentPerception + learning_rate*(currentReward - currentPerception)
    return newReward


def attendance_probability(estimate, temperature=10.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = np.asarray(estimate, dtype=float) / temperature
    return np.exp(-np.logaddexp(0, -z))