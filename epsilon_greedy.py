import numpy as np

def epsilon_greedy(delta, n, c, num_simulations):
    mu1 = 0
    mu2 = mu1 - delta
    Mu = np.array([mu1, mu2])
    rng = np.random.default_rng()
    K = len(Mu)
    regrets = np.zeros(num_simulations)

    for sim in range(num_simulations):
        empirical_means = np.ones(K)
        actions = []
        N = np.zeros(K)

        for i in range(1, n + 1):
            epsilon = np.min([1, c / i])
            b = np.random.binomial(n = 1, p = epsilon, size = 1)[0]
            
            # Explore
            if b == 1:
                selected_arm = rng.integers(0, K)
                reward = np.random.normal(loc = Mu[selected_arm], scale = 1, size = 1)[0]
                empirical_means[selected_arm] = ((empirical_means[selected_arm] * N[selected_arm]) + reward) / (N[selected_arm] + 1)
                actions.append(selected_arm)
                N[selected_arm] += 1
            # Exploit
            else:
                selected_arm = np.argmax(empirical_means)
                reward = np.random.normal(loc = Mu[selected_arm], scale = 1, size = 1)[0]
                empirical_means[selected_arm] = ((empirical_means[selected_arm] * N[selected_arm]) + reward) / (N[selected_arm] + 1)
                actions.append(selected_arm)
                N[selected_arm] += 1
        
        # Compute Regret
        regret_step = np.max(Mu) - Mu[actions]
        regret = np.sum(regret_step)
        regrets[sim] = regret

    average_regret = np.average(regrets)
    stderr = regrets.std(ddof=1) / np.sqrt(num_simulations)

    return average_regret, stderr