import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size
    
    return (X.T @ (sigmoid(X @ theta)-y)/m)

def gradient_descent(X, y , L=0.1, n_iters=1000, tolerance=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    
    for i in range(n_iters):
        grad = calculate_gradient(theta, X_b, y)
        theta -= L * grad
        
        if np.linalg.norm(grad) < tolerance:
            break
    
    return theta

def predict_proba(X, theta):
    #prepending the bias 
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold=0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

theta_hat = gradient_descent(X_train_scaled, y_train)
y_predictions_train = predict(X_train_scaled, theta_hat)
y_predictions_test = predict(X_test_scaled, theta_hat)

train_acc = accuracy_score(y_train, y_predictions_train)
test_acc = accuracy_score(y_test, y_predictions_test)

print(f'Train accuracy: {train_acc}')
print(f'Test accuracy: {train_acc}')