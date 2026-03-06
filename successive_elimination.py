import numpy as np

def successive_elimination(delta, n, num_simulations):
    mu1 = 0
    mu2 = mu1 - delta
    Mu = np.array([mu1, mu2])
    K = len(Mu)
    regrets = np.zeros(num_simulations)

    for sim in range(num_simulations):
        active = np.ones(K)
        N = np.zeros(K)
        empirical_means = np.zeros(K)
        UCB = np.zeros(K)
        LCB = np.zeros(K)
        actions = []
        t = 0

        while t < n:
            active_arms = np.where(active == 1)[0]

            for i in active_arms:
                actions.append(i)
                reward = np.random.normal(loc = Mu[i])
                empirical_means[i] = ((empirical_means[i] * N[i]) + reward) / (N[i] + 1)
                N[i] += 1
            
                bonus = np.sqrt((2 * np.log(n)) / N[i])
                UCB[i] = empirical_means[i] + bonus
                LCB[i] = empirical_means[i] - bonus

            # Deactivate arms where UCB_i(t) < LCB_i'(t)
            for i in active_arms:
                other_arms = active_arms[active_arms != i]

                if np.any(UCB[i] < LCB[other_arms]):
                    active[i] = 0
            
            # t += number of active arms
            t = t + len(active_arms)

        regret_step = np.max(Mu) - Mu[actions]
        regret = np.sum(regret_step)
        regrets[sim] = regret
    
    average_regret = np.average(regrets)
    stderr = regrets.std(ddof=1) / np.sqrt(num_simulations)

    return average_regret, stderr
