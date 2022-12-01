import numpy as np


class GeometricMean(object):
    def __init__(self):
        return

    def calculate(self, list1, list2):
        mean_list = []
        for el1, el2 in zip(list1, list2):
            if el1 == 0:
                mean_list.append(el2)
            elif el2 == 0:
                mean_list.append(el1)
            else:
                mean_list.append(np.sqrt(el1 * el2))
        return mean_list
