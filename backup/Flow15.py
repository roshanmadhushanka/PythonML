import random
import matplotlib.pyplot as plt
from featureeng import Math

lst = [5, 8, 3, 4, 8, 4, 7, 7, 0, 2, 1, 7, 8, 1, 3, 9, 7, 6, 9, 3, 7, 4, 0, 3, 3, 9, 7, 8, 2, 5, 9, 5, 2, 9, 4, 2, 4, 2, 6, 1, 5, 6, 4, 9, 2, 4, 3, 0, 2, 4, 2, 9, 0, 7, 6, 5, 2, 4, 5, 2, 5, 5, 7, 2, 1, 4, 0, 3, 3, 0, 3, 4, 8, 8, 2, 8, 2, 2, 3, 7, 4, 2, 7, 0, 9, 7, 3, 6, 3, 9, 7, 0, 3, 9, 6, 4, 9, 3, 3, 6]
probability = Math.moving_probability(series=lst, window=10, no_of_bins=4, default=False)
entropy = Math.moving_entropy(series=lst, window=10, no_of_bins=4, default=False)

index = range(len(probability))

plt.plot(index, probability)
plt.plot(index, entropy)

plt.legend(['probability', 'entropy'], loc='upper left')
plt.show()