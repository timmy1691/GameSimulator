import math
import numpy as np

def basicExpectedRewards(lastReward, perceivedReward, memoryWeight = 1):
    r_1 = lastReward
    q_1 = perceivedReward
    return (1- memoryWeight) * perceivedReward + memoryWeight*lastReward


def expectationUpdatebasic(currentPerception, currentReward, learning_rate):
    newReward = currentPerception + learning_rate*(currentReward - currentPerception)
    return newReward


def perceptionUpdateBasic(currentPerception, currentReward, learning_rate):
    """
    Perception is the estimated number of attemdees
    currentReward is the current normazlied rewards
    learning_rate is taking the difference, this will be the wieghted average, and how much weighting to new information
    """

    return currentPerception + learning_rate*currentReward

def perceptionUpdateBasic(perceived_reward, rewards, attendance, numPlayers, learning_rate=0.5):
    """
    Percieved Rewards'
    Actual Rewards
    Attendance
    numPlayers - population normalization 
    """
    
    
    updated = np.asarray(perceived_reward, dtype=float).copy()
    attended = np.asarray(attendance, dtype=bool)
    normalized_reward = np.asarray(rewards, dtype=float) / numPlayers

    updated[attended] = (
        (1 - learning_rate) * updated[attended]
        + learning_rate * normalized_reward[attended]
    )

    return updated

def attendance_probability(estimate, temperature=10.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z = np.asarray(estimate, dtype=float) / temperature
    return np.exp(-np.logaddexp(0, -z))