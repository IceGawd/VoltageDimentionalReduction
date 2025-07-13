import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def mutual_information(voltage_matrix, N=1000, num_samples=1000):
    """
    Approximate I(A; B) via Gaussian mixture and Monte Carlo sampling.

    for Details see: highLevelDocs/Mutual_Information/VoltageMutualInformation.pdf
    
    Parameters:
    - voltage_matrix: (n, K) array of voltages V_i(u) (n=dimension, k=number of voltages)
    - N: number of particles (high numbers cause MI to be close to the maximum entropy)
    - num_samples: number of Monte Carlo samples
    
    Returns:
    - Approximation of I(A; B)
    """
    n, K = voltage_matrix.shape
    V = voltage_matrix

    # confine V to be between 1./N and 1-1./N to avoid numerical issues due to gaussian approx of binomial
    V = np.clip(V, 1./N, 1- 1./N)
    
    # Compute means and covariances for each node
    means = N * V
    covs = np.array([
        np.diag(N * V[u] * (1 - V[u]))
        for u in range(n)
    ])

    # Compute conditional entropies H(A|B=u)
    H_cond_list = []
    log_2pi_e = np.log(2 * np.pi * np.e)
    for u in range(n):
        log_det = np.sum(np.log(N * V[u] * (1 - V[u])))
        H_u = 0.5 * (K * log_2pi_e + log_det)
        H_cond_list.append(H_u)
    H_cond = np.mean(H_cond_list)

    # Monte Carlo sampling to estimate H(A)
    np.random.seed(42)  # For reproducibility
    num_samples = min(num_samples, 1000000)  # Limit to avoid excessive memory usage
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    us = np.random.randint(n, size=num_samples)
    samples = np.array([np.random.multivariate_normal(means[u], covs[u]) for u in us])    
    
    # Estimate log-density at samples under mixture
    inv_covs = 1.0 / np.array([np.diag(cov) for cov in covs])  # (n, K)
    log_det_covs = np.sum(np.log([np.diag(cov) for cov in covs]), axis=1)  # (n,)
    
    log_probs = []
    for x in samples:
        diffs = means - x  # (n, K)
        mahal = np.sum(diffs**2 * inv_covs, axis=1)  # (n,)
        logpdfs = -0.5 * (K * np.log(2 * np.pi) + log_det_covs + mahal)
        logpdfs -= np.log(n)
        log_p_x = logsumexp(logpdfs)
        log_probs.append(log_p_x)
    H_marginal = -np.mean(log_probs)
    

    # Mutual information
    I_est = H_marginal - H_cond
    return I_est

if __name__ == "__main__":
    # test case

    epsilon=0.02
    for i in range(1, 10):
        voltage_matrix=np.eye(i)  # Example with i nodes and i sources
        # Estimate mutual information
        I_est = mutual_information(voltage_matrix,N=100000,num_samples=10000)
        print(f"i={i},Estimated I(A; B) ≈ {I_est:.4f} nats log({i})-I_est = {np.log(i) - I_est:.4f}")
        if np.abs(np.log(i) - I_est) > epsilon:
            raise ValueError(f"Test failed: difference between log({i}) and I_est is greater than {epsilon}")
    print(f"Test passed: I_est is within {epsilon} of log(i) for all i in [2, 3, ..., 9]")
