import numpy as np
from shapely.geometry import Point, Polygon


# def limited_area(bbox, area_poly):
#     makecenter = lambda x1, y1, x2, y2 : (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
#     bbox_center = [makecenter(box[0], box[1], box[2], box[3]) for box in bbox]
#     checkpoint = lambda x : x.within(area_poly)
#     check = np.array([checkpoint(Point(dot)) for dot in bbox_center])

#     return check


def check_meal_water(dot):
    if dot.within(ch1_meal_poly) or dot.within(ch2_meal_poly) or dot.within(ch3_meal_poly):
        meal = 1
    else:
        meal = 0

    if dot.within(ch1_water_poly) or dot.within(ch2_water_poly) or  dot.within(ch3_water_poly):
        water = 1
    else:
        water = 0

    return meal, water


ch1_meal = [
	(24,724),
	(1809,773),
	(1659,589),
	(12,555),
]

ch1_water = [
	(487,244),
	(1379,260),
	(1310,174),
	(487,167),
]

ch2_meal = [
	(25,952),
	(1832,918),
	(1704,771),
	(167,787),
]

ch2_water = [
	(498,394),
	(1362,375),
	(1308,317),
	(559,325),
]
ch3_meal = [
	(41,784),
	(1795,696),
	(1684,522),
	(157,625),
]
ch3_water = [
	(434,249),
	(1431,172),
	(1332,102),
	(504,151),
]


# ch1_shared_area = [
# ()
# ]
# ]

ch1_meal_poly = Polygon(ch1_meal)
ch1_water_poly = Polygon(ch1_water)
ch2_meal_poly = Polygon(ch2_meal)
ch2_water_poly = Polygon(ch2_water)
ch3_meal_poly = Polygon(ch3_meal)
ch3_water_poly = Polygon(ch3_water)