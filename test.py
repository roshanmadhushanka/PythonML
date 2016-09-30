from featureeng import Math
import random

lst = [4, 8, 5, 3, 5, 9, 9, 1, 8, 5, 7, 3, 9, 9, 1, 4, 3, 3, 7, 1, 6, 8, 4, 8, 4, 5, 3, 4, 8, 1, 4, 6, 1, 7, 5, 4, 7, 4, 2, 8, 6, 7, 5, 3, 3, 2, 6, 3, 8, 6, 9, 3, 4, 7, 7, 7, 0, 7, 4, 1, 2, 2, 6, 6, 2, 8, 6, 3, 8, 1, 8, 3, 7, 0, 2, 7, 0, 4, 8, 8, 6, 6, 9, 0, 1, 1, 8, 3, 9, 1, 9, 1, 7, 4, 4, 3, 9, 6, 7, 1]

arr = Math.moving_threshold_average(series=lst, window=5, threshold=1)

print arr