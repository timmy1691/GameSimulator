import numpy as np
import plotly.graph_objects as go

def farolBar(
        numPlayers,
        numRounds,
        maxReward = 6,
        seed = 42,
        strategy = "random"
):

    attendanceHistory = np.zeros((numRounds, numPlayers))
    rewardHistory = np.zeros((numRounds, numPlayers))

    for i in range(numRounds):
        attendance = np.random.randint(0, 2, size=(numPlayers))
        print("attendacne", attendance, flush=True)
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




if __name__ == "__main__" :
    randomRewards, randomScore = farolBar(100, 2)

    print("scores of random congestion game", randomScore)

    print("rewards of random congestion game", randomRewards)