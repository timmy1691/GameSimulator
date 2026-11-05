import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from GameSimulator.agentRewards import basicRewards as bR


def farolBarRandom(
        numPlayers,
        numRounds,
        maxReward = 6,
        seed = 42,
):
    """
    Playing the game if all of the players are doing random choices
    """

    attendanceHistory = np.zeros((numRounds, numPlayers))
    rewardHistory = np.zeros((numRounds, numPlayers))


    for i in range(numRounds):
        attendance = np.random.randint(0, 2, size=(numPlayers))
        totalAttendance = sum(attendance)

        if totalAttendance > maxReward:
            actualReward = maxReward - totalAttendance
        else:
            actualReward = attendance

        rewards = actualReward * attendance
        attendanceHistory[i, :] = attendance
        rewardHistory[i, :] = rewards

    
    aggScores = rewardHistory.sum(axis = 0)

    return rewardHistory, aggScores

def farolBarStrategyRandomWithUpdate(
        numPlayers,
        numRounds,
        maxReward = 6,
        seed = 42,
        percievedHistoryRandomness = 0.5,
        learning_rate = 0.5):

    attendanceHistory = np.zeros((numRounds, numPlayers))
    rewardHistory = np.zeros((numRounds, numPlayers))

    PercievedHistory = np.array([percievedHistoryRandomness] * numPlayers)

    PerceptionHistory = np.zeros((numRounds, numPlayers))

    for i in range(numRounds):
        attendance = np.zeros((numPlayers))
        for j in range(numPlayers):
            attendanceChoice = np.random.uniform(0, 1)
            if attendanceChoice <= PercievedHistory[j]:
                attendance[j] = 1
            else:
                attendance[j] = 0

        totalAttendance = sum(attendance)

        if totalAttendance > maxReward:
            actualReward = maxReward - totalAttendance
        else:
            actualReward = totalAttendance

        rewards = actualReward * attendance
        attendanceHistory[i, :] = attendance
        rewardHistory[i, :] = rewards

        PerceptionHistory[i, :] = PercievedHistory
        PercievedHistory = bR.perceptionUpdateBasic(PercievedHistory, rewards, learning_rate=learning_rate)

    return PercievedHistory, rewardHistory, attendanceHistory, PerceptionHistory


def farolBarStrategies(        
        numPlayers,
        numRounds,
        maxReward = 6,
        seed = 42,
        strategy = "random",
        playerPerceptions = None,
        playerProbabilities = None,
        learning_rate = 0.5

        ):

    perceptionHistory = np.zeros((numRounds, numPlayers))
    
    attendanceHistory = np.zeros((numRounds, numPlayers))
    rewardHistory = np.zeros((numRounds, numPlayers))
    if playerPerceptions is None:
        playerPerceptions = np.zeros((1, numPlayers))

    if playerProbabilities is None:
        playerProbabilities = [0.5]*numPlayers

    for i in range(numRounds):
        if i == 0:
            currentAttendance = np.random.randint(0, 2, size=(numPlayers))

            totalAttendance = sum(currentAttendance)

            if totalAttendance > maxReward:
                actualReward = maxReward - totalAttendance
            else:
                actualReward = totalAttendance

        else:
            currentAttendance = np.zeros((numPlayers))
            for j in range(numPlayers):
                attendanceChoice = np.random.uniform(0, 1)
                if attendanceChoice <= playerProbabilities[0, j]:
                    currentAttendance[j] = 1
                else:
                    currentAttendance[j] = 0
            totalAttendance = sum(currentAttendance)
            if totalAttendance > maxReward:
                actualReward = maxReward - totalAttendance
            else:
                actualReward = totalAttendance

        rewards = actualReward * currentAttendance
        attendanceHistory[i, :] = currentAttendance
        rewardHistory[i, :] = rewards
        playerPerceptions = bR.expectationUpdatebasic(playerPerceptions, rewards, learning_rate=learning_rate)
        playerProbabilities = bR.attendance_probability(playerPerceptions)

    return playerPerceptions, playerProbabilities, rewardHistory, attendanceHistory

