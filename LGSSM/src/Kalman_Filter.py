import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm



def compute_expectation(obs, xt, xtprev, t, N_particles):
        h_tilde = np.zeros((4,N_particles))
        h_tilde[0, :] = xt * obs[t]
        h_tilde[1, :] = xt**2
        if t == 0:
            h_tilde[2, :] = np.zeros(N_particles)
        else:
            h_tilde[2, :] = np.multiply(xt, xtprev)
        if t == 0:
            h_tilde[3, :] = np.zeros(N_particles)
        else:
            h_tilde[3, :] = xtprev**2
        return h_tilde


def exp_weight(log_weights):
    w = np.exp(log_weights - np.max(log_weights))
    w = w / np.sum(w)
    return w


class KalmanFilter:
    ''' KalmanFilter for state estimation and prediction in linear dynamic systems, including functions for filtering, smoothing, and likelihood computation.

    Parameters
    ----------
    A: The state transition matrix.
    B: The control input matrix.
    Q: The process noise covariance matrix.
    R: The measurement noise covariance matrix.

    Methods
    -------
    run_filter:
        Estimates the state and covariance at each time step using the Kalman Filter.

    run_smoother:
        Performs backward smoothing on the estimated states using future information.

    run_lag_one_covariance_smoother:
        Computes the lag-one covariance smoother, which represents the covariance between the current state estimate and the next state estimate 
        given the observations.

    run:
        Computes exact sufficient statistics, including the expected state, state squared, cross-covariance, and state covariance.
    '''

    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def run_filter(self, Y, T=None):
        if T is None:
            T = len(Y)

        x = 0.0
        P = 1.0

        self.filtered_state_means = np.zeros(T)
        self.filtered_state_covariances = np.ones(T)

        for t in range(T):
            y = Y[t]

            # Time Update
            x_pred = self.A * x
            P_pred = self.A * P * self.A + self.Q

            # Measurement Update
            K = P_pred * self.B / (self.B * P_pred * self.B + self.R)
            x = x_pred + K * (y - self.B * x_pred)
            P = (1 - K * self.B) * P_pred

            self.filtered_state_means[t] = x
            self.filtered_state_covariances[t] = P

        return self.filtered_state_means, self.filtered_state_covariances


    def run_smoother(self, Y, T=None):
        if T is None:
            T = len(Y)

        self.smoothed_state_means = self.filtered_state_means.copy()
        self.smoothed_state_covariances = self.filtered_state_covariances.copy()

        for t in reversed(range(T - 1)):
            P_inter = self.A * self.smoothed_state_covariances[t] * self.A + self.Q
            L = self.smoothed_state_covariances[t] * self.A / P_inter
            self.smoothed_state_means[t] += L * (self.smoothed_state_means[t+1] - self.A * self.smoothed_state_means[t])
            self.smoothed_state_covariances[t] += L * (self.smoothed_state_covariances[t+1] - P_inter) * L

        return self.smoothed_state_means, self.smoothed_state_covariances


    def run_lag_one_covariance_smoother(self, Y, T=None):
        if T is None:
            T = len(Y)

        if T == 1:
          return np.zeros(1)

        filtered_state_covariances = self.filtered_state_covariances
        lag_one_smoother = np.zeros(T-1)
        P_inter = self.A * filtered_state_covariances[-2] * self.A + self.Q
        K = P_inter * self.B / (self.B * P_inter * self.B + self.R)
        lag_one_smoother[-1] = (1 - K * self.B)*self.A*filtered_state_covariances[-2]

        for t in reversed(range(T - 2)):
            P_inter1 = self.A * filtered_state_covariances[t+1] * self.A + self.Q
            P_inter2 = self.A * filtered_state_covariances[t] * self.A + self.Q
            L = filtered_state_covariances[t+1] * self.A / P_inter1
            LL = filtered_state_covariances[t] * self.A / P_inter2
            lag_one_smoother[t] = filtered_state_covariances[t+1]*LL + L*(lag_one_smoother[t+1] - self.A*filtered_state_covariances[t+1])*LL

        self.lag_one_smoother = np.insert(lag_one_smoother,0,0)

        return self.lag_one_smoother

    def run(self, Y):

        T = len(Y)
        exact_sufficient_statistics = np.zeros((4, T))
        for t in range(1, T+1):
            filtered_state_means, filtered_state_covariances = self.run_filter(Y, t)
            smoothed_state_means, smoothed_state_covariances = self.run_smoother(Y, t)
            lag_one_smoother = self.run_lag_one_covariance_smoother(Y, t)
            smoothed_state_means = smoothed_state_means.reshape(-1)
            current_sufficient_statistics = np.zeros((4, t))
            current_sufficient_statistics[0, 0] = smoothed_state_means[0]*Y[0]
            current_sufficient_statistics[1, 0] = smoothed_state_means[0]**2
            for tt in range(1, t):
                current_sufficient_statistics[:, tt] = current_sufficient_statistics[:, tt-1] + compute_expectation(Y, smoothed_state_means[tt], smoothed_state_means[tt-1], tt, 1).reshape(4)
                current_sufficient_statistics[1, tt] += smoothed_state_covariances[tt]
                current_sufficient_statistics[2, tt] += lag_one_smoother[tt]
                current_sufficient_statistics[3, tt] += smoothed_state_covariances[tt-1]
            exact_sufficient_statistics[:, t-1] = current_sufficient_statistics[:, -1]
            if t == T:
                self.lik = np.zeros(T)
                for tt in range(1, T):
                    self.lik[tt] = smoothed_state_means[tt]  
        self.filtMean = exact_sufficient_statistics


    def run2(self, Y):

        T = len(Y)

        filtered_state_means, filtered_state_covariances = self.run_filter(Y)
        smoothed_state_means, smoothed_state_covariances = self.run_smoother(Y)
        lag_one_smoother = self.run_lag_one_covariance_smoother(Y)

        self.lik = np.zeros((4, T))
        self.lik[0, 0] = smoothed_state_means[0]
        self.lik[1, 0] = smoothed_state_means[0]**2
        for t in range(1, T):
            self.lik[0, t] = smoothed_state_means[t]
            self.lik[1, t] = smoothed_state_means[t]**2 + smoothed_state_covariances[t]
            self.lik[2, t] += smoothed_state_means[t-1]*smoothed_state_means[t] + lag_one_smoother[t]
            self.lik[3, t] += smoothed_state_means[t-1]**2 + smoothed_state_covariances[t-1]