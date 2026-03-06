import numpy as np

def ETC(delta, n, num_simulations):
    mu1 = 0
    mu2 = mu1 - delta
    Mu = np.array([mu1, mu2])
    M = int(np.ceil(n ** (2/3)))
    K = len(Mu)
    regrets = np.zeros(num_simulations)

    for sim in range(num_simulations):
        # Exploration
        exploration_samples = np.random.normal(loc = Mu[:, None], scale = 1, size = (K, M))
        empirical_means = np.mean(exploration_samples, axis = 1)
        exploration_actions = np.arange(K).repeat(M)

        # Exploitation
        i_hat = np.argmax(empirical_means)
        exploitation_actions = np.full(n - (M * K), i_hat)
        action_array = np.concatenate((exploration_actions, exploitation_actions))

        # Compute Regret
        regret_step = np.max(Mu) - Mu[action_array]
        regret = np.sum(regret_step)
        regrets[sim] = regret

    # Calculate average regret + stderr
    average_regret = np.average(regrets)
    stderr = regrets.std(ddof=1) / np.sqrt(num_simulations)

    return average_regret, stderr