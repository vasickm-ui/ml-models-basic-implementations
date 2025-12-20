import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/students_grades.csv')

# print(data.columns)
# plt.scatter(data.time, data.score)
# plt.xlabel("Time")
# plt.ylabel("Score")
# plt.title("Students scores in relation with time")
# plt.show()

def loss_function(m,b,points) -> float:
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].time
        y = points.iloc[i].score
        total_error += (y - m*x - b)**2
    return total_error/float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].time
        y = points.iloc[i].score
        
        m_gradient += -(2/n) * x * (y - m_now * x - b_now)
        b_gradient += -(2/n)  * (y - m_now * x - b_now)
        
    m = m_now - learning_rate * m_gradient
    b = b_now - learning_rate * m_gradient
    return m, b

m , b = 0, 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    m,b = gradient_descent(m,b,data,L)
    
print(m,b)
    
plt.scatter(data.time, data.score)
plt.plot(list(range(60)), [m*x+b for x in range(60)], color = "red")
plt.show()
    
        

