from collections import namedtuple
from itertools import product
from math import sqrt
from pprint import pprint as pp
import matplotlib.pyplot as plt


def covers(cir, pt):
    return (cir[0] - pt[0])**2 + (cir[1] - pt[1])**2 <= cir[2]**2

def circles(p1, p2, r):
	if p1 == p2:
		return None, None
    # delta x, delta y between points
	dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # dist between points
	q = sqrt(dx**2 + dy**2)
    #separation of points > diameter
	if q > 2.0*r:
		return None, None
    # halfway point
	x3, y3 = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    # distance along the mirror line
	d = sqrt(r**2-(q/2)**2)
    # Circle one
	c1 = (x3 - d*dy/q,y3 + d*dx/q,abs(r))
	# Circle two
	c2 = (x3 + d*dy/q,y3 - d*dx/q,abs(r))
	return c1, c2

def min_circles(points,r):
    #points=[(-5, 5), (-4, 4), (3, 2), (1, -1),(0,0),(-2,-4),(0,4),(-1,4), (4, 4), (2, 4), (-3, 2),(4,2), (6,-6)]
    n, p = len(points), points
    # All circles between two points (which can both be the same point)
    circls = set(sum([[c1, c2]
                    for c1, c2 in[circles(p1, p2, r) for p1, p2 in product(p, p)]
                	if c1 is not None], []))
    # points covered by each circle 
    coverage = {c: {pt for pt in points if covers(c, pt)}
                for c in circls}
    # Ignore all but one of circles covering points covered in whole by other circles
    items = sorted(coverage.items(), key=lambda keyval:len(keyval[1]))
    for i, (ci, coveri) in enumerate(items):
        for j in range(i+1, len(items)):
            cj, coverj = items[j]
            if not coverj - coveri:
                coverage[cj] = {}
    coverage = {key: val for key, val in coverage.items() if val}
    #print("cov",coverage)
    # Greedy coverage choice 
    chosen, covered = [], set()

    while len(covered) < n:
        _, nxt_circle, nxt_cov = max((len(pts - covered), c, pts)
                                     for c, pts in coverage.items())
        delta = nxt_cov - covered
        covered |= nxt_cov # covered = covered | nxt_cov
        chosen.append([nxt_circle, delta])
    CN=len(chosen)
    return points,chosen,CN


#"""""""""""""""""""""""""""""""""""""""""""TEST"""""""""""""""""""""""""""""""""""""""""#

figure, axes = plt.subplots()
points = [(-5, 5), (-4, 4), (3, 2), (1, -1),(0,0),(-2,-4),(0,4),(-1,0),(4,4),(2,4), (-3, 2), (4, -2), (6, -6)]
rayon=3
pts,circles,CN=min_circles(points,rayon)
print("Covering number = ",CN)


n=len(points)
print('\n%i points'%n)
print(pts)
print('A minimum of circles of radius %g to cover the points (And the extra points they covered)'%rayon)
print("\n",circles)


for pts in points:
    axes.scatter(pts[0],pts[1], c ="pink",
        marker ="o",
        edgecolor ="green",
        s = 50)
for c in circles: 
    Drawing_uncolored_circle=plt.Circle(( c[0][0],c[0][1] ), 3 ,fill = False )
    axes.add_artist( Drawing_uncolored_circle)
plt.show()