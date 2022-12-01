class Sum(object):
    def __init__(self):
        return

    def calculate(self, list1, list2):
        uncertainties = [l1 + l2 for l1, l2 in zip(list1, list2)]
        return uncertainties


class Multiply(object):
    def __init__(self):
        return

    def calculate(self, list1, list2):
        uncertainties = [l1 * l2 for l1, l2 in zip(list1, list2)]
        return uncertainties
