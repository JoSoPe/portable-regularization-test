import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Gradient Descent - No Regularization
def gradient_descent(X, y, theta, lr, iterations):
    m = y.shape[0]
    start = time.time()
    cost_history = []
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (2/m) * X.T @ errors
        theta -= lr * gradient
        cost = (1/m) * torch.sum(errors**2).item()
        cost_history.append(cost)
    return time.time() - start, cost_history

# L1 Regularization
def gradient_descent_l1(X, y, theta, lr, iterations, l1_penalty):
    m = y.shape[0]
    start = time.time()
    cost_history = []
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (2/m) * X.T @ errors + (l1_penalty/m) * torch.sign(theta)
        theta -= lr * gradient
        cost = (1/m) * torch.sum(errors**2).item() + (l1_penalty/m) * torch.sum(torch.abs(theta)).item()
        cost_history.append(cost)
    return time.time() - start, cost_history

# L2 Regularization
def gradient_descent_l2(X, y, theta, lr, iterations, l2_penalty):
    m = y.shape[0]
    start = time.time()
    cost_history = []
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (2/m) * X.T @ errors + (2*l2_penalty/m) * theta
        theta -= lr * gradient
        cost = (1/m) * torch.sum(errors**2).item() + (l2_penalty/m) * torch.sum(theta**2).item()
        cost_history.append(cost)
    return time.time() - start, cost_history

# Main
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, dtype="float32")

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X.insert(0, 'Intercept', 1.0)

    # Hyperparameters
    lr = 0.01
    iterations = 1000
    display_iterations = 200
    l1_penalty = 0.1
    l2_penalty = 0.1

    results = []

    for device in [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []):
        device_name = "GPU" if device.type == "cuda" else "CPU"
        print(f"\nRunning computations on {device_name}...")

        # Convert to PyTorch tensors on the selected device
        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32, device=device)
        theta_init = torch.zeros(X_tensor.shape[1], dtype=torch.float32, device=device)

        time_gd, cost_history = gradient_descent(X_tensor, y_tensor, theta_init.clone(), lr, iterations)
        time_l1, cost_history_l1 = gradient_descent_l1(X_tensor, y_tensor, theta_init.clone(), lr, iterations, l1_penalty)
        time_l2, cost_history_l2 = gradient_descent_l2(X_tensor, y_tensor, theta_init.clone(), lr, iterations, l2_penalty)

        results.append({
            "device": device_name,
            "time_gd": time_gd,
            "time_l1": time_l1,
            "time_l2": time_l2,
            "cost_history": cost_history,
            "cost_history_l1": cost_history_l1,
            "cost_history_l2": cost_history_l2
        })

    # Combined plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Bar plot for computation time
    for result in results:
        methods = [f'{m} ({result["device"]})' for m in ['GD', 'L1 GD', 'L2 GD']]
        times = [result['time_gd'], result['time_l1'], result['time_l2']]
        axs[0].bar(methods, times, label=result['device'])

    axs[0].set_ylabel('Computation Time (sec)')
    axs[0].set_title('Gradient Descent Time Comparison (CPU vs GPU)')
    axs[0].grid(axis='y')
    axs[0].legend()

    # Line plot for cost function history
    for result in results:
        axs[1].plot(range(display_iterations), result['cost_history'][:display_iterations], label=f'GD ({result["device"]})')
        axs[1].plot(range(display_iterations), result['cost_history_l1'][:display_iterations], label=f'L1 GD ({result["device"]})')
        axs[1].plot(range(display_iterations), result['cost_history_l2'][:display_iterations], label=f'L2 GD ({result["device"]})')

    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Cost')
    axs[1].set_title(f'Cost Function History Comparison (First {display_iterations} Iterations)')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()