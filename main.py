import time
import matplotlib.pyplot as plt
from itertools import combinations
import math
import numpy as np
import random

import test

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

def form_circle(points):
    if len(points) == 0:
        return Circle((0, 0), 0)
    
    elif len(points) == 1:
        return Circle(points[0], 0)
    
    elif len(points) == 2:
        midpoint = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        r = math.dist(points[0], points[1]) / 2

        return Circle(midpoint, r)
    
    elif len(points) == 3:
        # COPIED FROM: https://stackoverflow.com/questions/28910718/given-three-points-find-the-center-of-circle
        p1, p2, p3 = points[0], points[1], points[2]

        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        # if the points are collinear
        if abs(det) < 1.0e-6:
            return Circle(None, np.inf)
        
        # center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        
        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

        return Circle((cx, cy), radius)

    else:
        return Circle(None, None)

def brute_force(points, pairs, triplets):
    # O(n)
    def encloses_points(circle):
        for point in points:
            if math.dist(circle.center, point) > circle.radius:
                return False
            
        return True

    smallest_circle = Circle(None, np.inf)

    # O(n^3)
    for pair in pairs:
        circle = form_circle(pair)

        if encloses_points(circle) and circle.radius < smallest_circle.radius:
            smallest_circle = circle

    # O(n^4)
    for triplet in triplets:
        circle = form_circle(triplet)

        if circle.center == None:
            continue

        if encloses_points(circle) and circle.radius < smallest_circle.radius:
            smallest_circle = circle
    
    return smallest_circle

def convex_hull(points, pairs, triplets):
    def counterclockwise(p1, p2, p3):
        # cross product (COPIED FROM: https://www.geeksforgeeks.org/orientation-3-ordered-points/)
        slope1 = (p2[1] - p1[1]) * (p3[0] - p2[0])
        slope2 = (p3[1] - p2[1]) * (p2[0] - p1[0])

        return slope2 > slope1

    def graham_scan():
        lowest_point = (0, 0)

        # O(n)
        for point in points:
            # if smaller y value, set as lowest point
            if point[1] < lowest_point[1]:
                lowest_point = point
            
            # if the current point and the lowest point share the same y value,
            # then check which one has the smaller x value
            elif point[1] == lowest_point[1]:
                if point[0] < lowest_point[0]:
                    lowest_point = point

        # O(nlogn) TimSort
        sorted_points = sorted(points, key=lambda point: math.atan2(point[1] - lowest_point[1], point[0] - lowest_point[0]))

        hull = []

        idx = 0

        # O(n)
        while idx < len(sorted_points):
            point = sorted_points[idx]

            # if stack is to small for any orientation stuff, add the point
            if len(hull) < 2:
                hull.append(point)
                idx += 1
                continue

            # orientation of the top 3 points in the stack
            o = counterclockwise(hull[-2], hull[-1], point)

            if o:
                hull.append(point)
                idx += 1
            
            else:
                hull.pop()
                continue

        return hull

    hull = graham_scan()
    circle = brute_force(hull, pairs, triplets)

    return circle, hull

def welzls(points, bounds):
    points = points.copy() # create a copy of the points for this call to avoid modifying the original list

    # recursion base case
    if len(points) == 0 or len(bounds) == 3:
        return form_circle(bounds)

    # remove a random point from points
    random_point = points.pop(random.randrange(len(points)))

    # call welzls on the remaining points
    circle = welzls(points, bounds.copy())

    # if the circle encloses the random point, return the circle
    if math.dist(circle.center, random_point) <= circle.radius:
        return circle
    
    # otherwise, call welzls on the remaining points with the random point added to bounds
    bounds.append(random_point)

    return welzls(points, bounds.copy())

# algorithm demo
# points = [
#     (random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(400)
# ]

# print(points)

points = test.test_points

# print times of all algorithms
pairs = combinations(points, 2)
triplets = combinations(points, 3)

start = time.time()
bc, bh = brute_force(points, pairs, triplets), []
length = time.time() - start

print("Brute Force Duration:", length)

pairs = combinations(points, 2)
triplets = combinations(points, 3)

start = time.time()
cc, ch = convex_hull(points, pairs, triplets)
length = time.time() - start

print("Convex Hull Duration:", length)

start = time.time() * 10000000
wc, wh = welzls(points, []), []
length = time.time() * 10000000 - start

print("Welzl's Duration (micro seconds):", length)


circle, hull = wc, wh

print("---- Demo ----")
print("Center:", circle.center, "Radius:", circle.radius)

# plotting
fig, ax = plt.subplots()

ax.set_aspect('equal', adjustable='datalim')

circle = plt.Circle(circle.center, circle.radius, color='g', fill=False)

ax.add_artist(circle)

ax.plot(
    [point[0] for point in points], 
    [point[1] for point in points], 
    'o'
)

if hull:
    ax.plot(
        [point[0] for point in hull] + [hull[0][0]], # add the first point to close the loop
        [point[1] for point in hull] + [hull[0][1]], # add the first point to close the loop
        'r-'
    )

plt.show()
