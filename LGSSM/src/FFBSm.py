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


class ForwardOnlyFFBSm:
    def __init__(self, A, B, Q, R, N_particles):
        ''' forward-only implemêntation Forward Filtering Backward Smoothing (FFBSm) for state estimation .

        Parameters
        ----------
        A: The state transition matrix.
        B: The control input matrix.
        Q: The process noise covariance matrix.
        R: The measurement noise covariance matrix.
        N_particles: The number of particles to use in the particle filter.

        Methods
        -------
        run:
            Executes the forward-only FFBSm algorithm to estimate the state given a sequence of observations.         
        '''

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N_particles = N_particles

    def run(self, obs):

        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        N_particles = self.N_particles
        T = len(obs)

        yt = obs[0]

        self.filtMean = np.zeros((4, T))

        # Initialize particles
        x = np.random.normal(0, 1, size=N_particles)
        log_weights = norm.logpdf(yt, B * x, np.sqrt(R)*np.ones_like(x))

        sufficient_statistics = compute_expectation(obs, x, None, 0, N_particles)
        self.filtMean[:, 0] = np.average(sufficient_statistics, weights=exp_weight(log_weights), axis=1)


        #for t in tqdm(range(1, T)):
        for t in range(1, T):
            yt = obs[t]

            # resample
            indX = np.random.choice(N_particles, size=N_particles, replace=True, p=exp_weight(log_weights))

            # Propagate and weight
            x_new = A*x[indX] + np.random.normal(0, np.sqrt(Q), size=N_particles)
            log_weights_new = norm.logpdf(yt, B * x_new, np.sqrt(R)*np.ones_like(x_new))

            # update the statistics with forward-only FFBSm update
            sufficient_statistics_new = np.random.randn(*sufficient_statistics.shape)
            for l in range(N_particles):

                backProbs = np.exp(log_weights + norm.logpdf(x_new[l], A * x, np.sqrt(Q)) + norm.logpdf(yt, B * x_new[l], np.sqrt(R)) )
                sufficient_statistics_new[:, l] = np.average(sufficient_statistics + compute_expectation(obs, x_new[l], x, t, N_particles), weights=backProbs, axis=1)

            sufficient_statistics = sufficient_statistics_new

            # compute the estimate
            self.filtMean[:, t] = np.average(sufficient_statistics, weights=exp_weight(log_weights_new), axis=1)

            # update particles and weights
            x = x_new
            log_weights = log_weights_new