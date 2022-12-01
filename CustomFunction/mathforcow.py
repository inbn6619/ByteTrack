import math
import numpy as np




def make_center(corr):
    xm, ym, xM, yM = corr
    xc = xm + (xM - xm) / 2
    yc = ym + (yM - ym) / 2

    return [xc, yc]

def find_distance(corr1, corr2):
    c1 = make_center(corr1)
    c2 = make_center(corr2)
    distance = math.sqrt(sum([i ** 2 for i in [abs(x) - abs(y) for x, y in zip(c1, c2)]]))

    return distance


def distance(corr1, corr2):
    c1, c2 = corr1[0], corr2[0]
    distance = math.sqrt(sum([i ** 2 for i in [abs(x) - abs(y) for x, y in zip(c1, c2)]]))

    return distance