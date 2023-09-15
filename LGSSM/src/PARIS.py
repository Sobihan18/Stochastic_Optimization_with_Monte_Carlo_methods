import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


class PARIS:
    def __init__(self, A, B, Q, R, N_particles=100, M=2):
        ''' Particle Rapid Incremental Smoother (PARIS) for state estimation .

        Parameters
        ----------
        A: The state transition matrix.
        B: The control input matrix.
        Q: The process noise covariance matrix.
        R: The measurement noise covariance matrix.
        N_particles: The number of particles to use in the particle filter.
        M: precision parameter

        Methods
        -------
        run:
            Executes the PARIS algorithm to estimate the state given a sequence of observations.         
        '''

        self.A = A
        self.B = B  
        self.Q = Q
        self.R = R
        self.N_particles = N_particles
        self.M = M

    def run(self, obs):

        A = self.A
        B = self.B
        Q = self.Q
        R = self.R
        N_particles = self.N_particles
        M = self.M
        T = len(obs)

        yt = obs[0]

        self.filtMean = np.zeros((4, T))

        # Initialize particles
        x = np.random.normal(0, 1, size=N_particles)
        log_weights = norm.logpdf(yt, B * x, np.sqrt(R)*np.ones_like(x))

        sufficient_statistics = compute_expectation(obs, x, None, 0, N_particles)
        self.filtMean[:, 0] = np.average(sufficient_statistics, weights=exp_weight(log_weights), axis=1)


        for t in range(1, T):
            yt = obs[t]

            # resample
            indX = np.random.choice(N_particles, size=N_particles, replace=True, p=exp_weight(log_weights))

            # Propagate and weight
            x_new = A*x[indX] + np.random.normal(0, np.sqrt(Q), size=N_particles)
            log_weights_new = norm.logpdf(yt, B * x_new, np.sqrt(R)*np.ones_like(x_new))


            # for each new particle compute backward probabilities and sample a backward index
            idx_M = np.zeros((N_particles, M), dtype='int')
            for l in range(N_particles):
                probs = log_weights_new + norm.logpdf(x_new[l], A * x, np.sqrt(Q)) + norm.logpdf(obs[t], B * x_new[l], np.sqrt(R))
                idx_M[l, :] = np.random.choice(N_particles, size=M, p=exp_weight(probs))


            # update the statistics with PaRIS update
            sufficient_statistics_inter = np.random.randn(*sufficient_statistics.shape)
            for j in range(M):
                sufficient_statistics_inter += sufficient_statistics[:, idx_M[:, j]] + compute_expectation(obs, x_new, x[idx_M[:, j]], t, N_particles)
            sufficient_statistics = sufficient_statistics_inter/M

            # compute the estimate
            self.filtMean[:, t] = np.average(sufficient_statistics, weights=exp_weight(log_weights_new), axis=1)

            # update particles and weights
            x = x_new
            log_weights = log_weights_new