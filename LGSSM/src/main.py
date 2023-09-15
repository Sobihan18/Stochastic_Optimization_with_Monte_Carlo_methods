import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from Kalman_Filter import *
from FFBSm import *
from PARIS import *


# Set random seed for reproducibility
np.random.seed(0)

# Generate example data
T = 100

# True parameters
A = 0.7
B = 1
Q = 0.1**2
R = 0.1**2


def compute_grad(model_name, obs, A, B, Q, R, N_particles = 100, M = 2, compute_likelihood=False):
    if model_name == 'Kalman':
        model = KalmanFilter(A, B, Q, R)
    if model_name == 'FFBSm':
        model = ForwardOnlyFFBSm(A, B, Q, R, N_particles)
    if model_name == 'PARIS':
        model = PARIS(A, B, Q, R, N_particles, M)
    model.run(obs)
    gradA = (1/Q) * (model.filtMean[2, -1] - A * model.filtMean[3, -1])
    gradB = (1/R) * (model.filtMean[0, -1] - B * model.filtMean[1, -1])
    if compute_likelihood:
        likelihood = - (1/2*Q)*(model.filtMean[1, -1] - 2*A*model.filtMean[2, -1] + A**2*model.filtMean[3, -1])
        likelihood -= (1/2*R)*(np.sum(obs**2) - 2*B*model.filtMean[0, -1] + B**2*model.filtMean[1, -1])
        return gradA, gradB, likelihood
    return gradA, gradB




if __name__ == "__main__":
    # Data Generation

    # Simulate the hidden states
    hidden_states = np.zeros(T)
    hidden_states[0] = np.random.normal(0, 1)
    for t in range(1, T):
        hidden_states[t] = A * hidden_states[t-1] + np.random.normal(0, np.sqrt(Q))

    # Simulate the observations
    obs = np.zeros(T)
    for t in range(T):
        obs[t] = B * hidden_states[t] + np.random.normal(0, np.sqrt(R))

    A_est_values2 = {'1/4' : [], '1/3' : [], '1/2' : [], '2/3' : [], '3/4' : [], '1' : [], 'c1' : [], 'c2' : [], 'c3' : [], 'c5' : [], 'c10' : []}
    loss_values2 = {'1/4' : [], '1/3' : [], '1/2' : [], '2/3' : [], '3/4' : [], '1' : [], 'c1' : [], 'c2' : [], 'c3' : [], 'c5' : [], 'c10' : []}
    biasA_values2 = {'1/4' : [], '1/3' : [], '1/2' : [], '2/3' : [], '3/4' : [], '1' : [], 'c1' : [], 'c2' : [], 'c3' : [], 'c5' : [], 'c10' : []}
    true_gradA_values2 = {'1/4' : [], '1/3' : [], '1/2' : [], '2/3' : [], '3/4' : [], '1' : [], 'c1' : [], 'c2' : [], 'c3' : [], 'c5' : [], 'c10' : []}
    gradA_values2 = {'1/4' : [], '1/3' : [], '1/2' : [], '2/3' : [], '3/4' : [], '1' : [], 'c1' : [], 'c2' : [], 'c3' : [], 'c5' : [], 'c10' : []}

    algo = 'Adagrad'
    model_name = 'FFBSm'
    #bias_terms = ['1/4', '1/3', '1/2' , '2/3', '3/4', '1']
    #bias_terms = ['c2', 'c3', 'c5', 'c10']
    bias_terms = ['1/2']

    epsilon = 1e-8      # Small constant to prevent division by zero
    epochs = 5
    C_gamma = 0.055

    for bias in bias_terms:
        # Initialization
        A_est = 0.4

        # Initialize accumulators for Adagrad
        gradA_sum = 0
        print(bias)
        for epoch in tqdm(range(epochs)):
            # compute true gradients and estimated gradients
            true_gradA, true_gradB = compute_grad('Kalman', obs, A_est, B, Q, R)
            if bias == 'constant':
                b = 5
            if bias == 'c1':
                b = 1
            if bias == 'c2':
                b = 2
            if bias == 'c3':
                b = 3
            if bias == 'c5':
                b = 5
            if bias == 'c10':
                b = 10
            elif bias == '1/4':
                b = int(np.ceil(np.sqrt(epoch+1)))
            elif bias == '1/3':
                b = int(np.ceil(np.power(epoch+1, 2/3)))
            elif bias == '1/2':
                b = epoch+1
            elif bias == '2/3':
                b = int(np.ceil(np.power(epoch+1, 4/3)))
            elif bias == '3/4':
                b = int(np.ceil(np.power(epoch+1, 6/4)))
            elif bias == '1':
                b = (epoch+1)**2

            print(bias)
            gradA, gradB, likelihood = compute_grad(model_name, obs, A_est, B, Q, R, N_particles = b + 1, M = (epoch+1) + 5, compute_likelihood=True)

            # Accumulate squared gradients
            gradA_sum += gradA**2

            # Learning rate schedule
            lr = C_gamma/np.sqrt(epoch+1)

            # Update parameters using Adagrad
            A_est += (lr / np.sqrt(gradA_sum + epsilon)) * gradA

            # Store the values
            A_est_values2[bias].append(A_est)
            loss_values2[bias].append(likelihood)
            biasA_values2[bias].append(np.abs(gradA - true_gradA))
            true_gradA_values2[bias].append(true_gradA)
            gradA_values2[bias].append(gradA)


            # Clear the previous plot
            plt.clf()

            plt.plot(range(epoch+1), A_est_values2[bias], label='Estimated A Values', color='red')
            plt.axhline(A, color='red', linestyle='--', label='True A')
            plt.xlabel('Epoch')
            plt.ylabel('Values')
            plt.title('Parameters')
            plt.legend()
            plt.show()
            plt.pause(0.01)  # Pause briefly to update the plot


            plt.plot(range(epoch+1), loss_values2[bias])
            plt.xlabel('Epoch')
            plt.ylabel('Values')
            plt.title('Likelihood')
            plt.show()

    # Keep the final plot displayed
    plt.show()