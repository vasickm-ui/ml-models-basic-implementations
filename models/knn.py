import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {"blue":[[1,1], [1,2], [1,3], [2,2], [2,3], [4,2], [5,3]],
          "red":[[3,1], [4,1], [4,3], [4,4], [5,2], [5,4], [1,3]]}

new_point = [2,4]
d = np.sqrt(np.sum((np.array([1,1])-np.array([3,2]))**2))
print(d)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

class KNN():
    def __init__(self, k):
        self.k = k
        self.point = None
    
    def fit(self, points):
        self.points = points
    
    def predict(self, new_point):
        distances = []
        
        for category in self.points:
            for point in self.points[category]:
                d = euclidean_distance(point, new_point)
                distances.append([d, category])
                
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
    
clf = KNN(3)
clf.fit(points) 

ax = plt.subplot()
ax.grid(True, color="grey")
ax.tick_params(axis='x', color='white')
ax.tick_params(axis='y', color='white')

for p in points['blue']:
    ax.scatter(p[0], p[1], color="blue", s=60)
    
for p in points['red']:
    ax.scatter(p[0], p[1], color="red", s=60)
    
new_class = clf.predict(new_point)
color = "red" if new_class == "red" else "blue"
ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)
        
        
for p in points['blue']:
    ax.plot([new_point[0],p[0]], [new_point[1],p[1]], color="blue", linestyle="--", linewidth="1")
    
for p in points['red']:
    ax.plot([new_point[0],p[0]], [new_point[1],p[1]], color="red", linestyle="--", linewidth="1")
    
plt.show()

