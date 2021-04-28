import numpy as np

recall = [2179, 2136, 2094, 2042, 1985, 1918, 1839, 1748, 1612, 1480, 1339, 1163, 804]
wrong_recall = [2201, 1779, 1468, 1224, 1008, 789, 574, 441, 277, 199, 147, 97, 34]

recall = np.array(recall)
wrong_recall = np.array(wrong_recall)
print(recall/(recall+wrong_recall))