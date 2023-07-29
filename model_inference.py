import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax
import numpy
import scipy
import time

import utils

# numpyro.set_platform("gpu")

model_beta_parameters = {}
func_numeric_mapping = {}


# TODO: Change the model_parameters value to probability dist.
# TODO : The values of model_parameters need to be the value of the dist (for example [alpha,beta] for Beta Dist.)

# When inferring the AI Agent, and before sending the model_parameters, we need to sample
# from the dist and send those sampled values to the model_parameters of the AI Agent.
# What we are expecting to see is an increase in the acc. rewards.
# Order is -> 1. Run Model, 2. Gather logs / acc. reward, 3. Update Posterior (model_parameters) and repeat 1.,2.,3...

def min_state(state):
    min_s = [0,0,0,0,0,0,0,0,0]
    for i in range(3):
        for j in range(3):
            if state[i][j] == 'X':
                min_s[i*3 + j] = -1
            elif state[i][j] == 'O':
                min_s[i*3 + j] = 1
    return min_s

def parse_obs(obs_file):
    """
    Parses the Observation File, returns a Dictionary where: Key: (Function Number, Action Parameter), Value: List of
    observations (list composed of 1's and 0's - 1 indicate that State != Next State, and 0 else).
    Notice that it ignores actions that are in utils.IGNORE_ACTIONS. In our case - 'opponent_mark' is ignored.
    For example: obs_map = [
    ((0, 5), [1, 1, 0, 0, 0, 0, 0, 1]),
    ((0, 1), [1, 1, 1, 1]),
    ((0, 8), [1, 1, 1, 1, 1, 1]),
    ((0, 4), [1, 1, 1, 1, 1]),
    ((0, 7), [1, 1, 1, 1, 1, 1, 1]),
    ((0, 3), [1, 1, 1, 1, 1]),
    ((0, 6), [1, 1, 1, 1, 1, 1, 1]),
    ((0, 2), [1, 1, 1, 1, 1, 1, 1, 1]),
    ((0, 9), [1, 1, 1])]

    Where 0 is mapped to 'ai_mark' function in global func_numeric_mapping variable. This mapping exists because NumPyro
    MCMC expects numeric values when running inference.

    This Function also adds initial 'alpha' and 'beta' parameters to the global model_beta_parameters for every new
    func_action and action_param it encounters.
    for example - model_beta_parameters = {
    (0, 5): {'alpha': 1, 'beta': 1},
    (0, 1): {'alpha': 1, 'beta': 1} ... }
    """
    func_numeric_count = 0
    obs_map = {}
    obs_list = []
    with open(obs_file, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces and newlines
            line_list = eval(line)  # Convert the line string to a list
            obs_list.append(line_list)
    global model_beta_parameters, func_numeric_mapping
    for episode in obs_list:
        for (state, (func_name, action_param), next_state) in episode:
            if func_name in utils.IGNORE_ACTIONS:
                continue
            if func_name in func_numeric_mapping.keys():
                func_numeric = func_numeric_mapping[func_name]
            else:
                func_numeric = func_numeric_count
                func_numeric_mapping[func_name] = func_numeric_count
                func_numeric_count += 1
            obs_map.setdefault((func_numeric, action_param), []).append(1) if state != next_state \
                else obs_map.setdefault((func_numeric, action_param), []).append(0)
            if not (func_numeric, action_param) in model_beta_parameters:
                model_beta_parameters[(func_numeric, action_param)] = {'alpha': 1, 'beta': 1}

    result = obs_map.items()
    obs_map = list(result)
    return obs_map


def fill_post_model_params(posterior, prior):
    for action in prior.keys():
        if action not in posterior:
            posterior[action] = prior[action]


def ai_model(obs=None, nobs=None):
    """
    (Tic-Tac-Toe) AI Agent model with NumPyro.
    p is a dictionary mapping (function, action parameter) -> sample from Beta distribution using (alpha, beta).
    obs = [
    ((0, 5), [1, 1, 0, 0, 0, 0, 0, 1]),
    ((0, 1), [1, 1, 1, 1]),
    ((0, 8), [1, 1, 1, 1, 1, 1]),
    ((0, 4), [1, 1, 1, 1, 1]),
    ((0, 7), [1, 1, 1, 1, 1, 1, 1]),
    ((0, 3), [1, 1, 1, 1, 1]),
    ((0, 6), [1, 1, 1, 1, 1, 1, 1]),
    ((0, 2), [1, 1, 1, 1, 1, 1, 1, 1]),
    ((0, 9), [1, 1, 1])]
    """
    # p ~ Beta(alpha, beta)
    global model_beta_parameters
    p = {}
    for (func, action_param) in model_beta_parameters.keys():
        alpha = model_beta_parameters[(func, action_param)]['alpha']
        beta = model_beta_parameters[(func, action_param)]['beta']
        p[(func, action_param)] = numpyro.sample(f"p({str(func)}, {str(action_param)})", dist.Beta(alpha, beta))
    for (func, action_param), p_i in p.items():
        if obs is not None:
            action_obs = [sublist[1] for sublist in obs if sublist[0] == (func, action_param)][0]
            nobs = len(action_obs)
        else:
            action_obs = None
            nobs = nobs
        nobs_i = nobs[(func, action_param)]
        with numpyro.plate("obs", size=nobs_i):
            numpyro.sample(f"o({str(func)}, {str(action_param)})", dist.Bernoulli(p_i),
                           obs=action_obs)


def prior_predictive(obs):
    """
    Evaluates Prior Predictive from the Prior Dist.
    Expecting to see mass in the bar where observation and imaginations are aligned.
    """
    prior_predi = numpyro.infer.Predictive(ai_model, num_samples=10000)
    prior_samples = prior_predi(jax.random.PRNGKey(int(time.time() * 1E6)),
                                nobs={key: len(value) for (key, value) in obs})
    if utils.PLOT:
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("prior predictive")
        for i in range(3):
            for j in range(3):
                ((func, action_param), action_obs) = obs[i % 3 + j % 3]
                key = f"o({str(func)}, {str(action_param)})"
                axs[i, j].set_title(f"Cell: {i * 3 + j + 1}")
                axs[i, j].set_xlim(-1, len(action_obs) + 1)
                axs[i, j].hist([sum(o) for o in prior_samples[key]], density=True, bins=len(action_obs) * 2 + 1,
                               label="imaginations")
                axs[i, j].axvline(sum(action_obs), color="red", lw=2, label="observation")
                axs[i, j].legend()
        plt.show()
    return prior_predi


def inference(obs):
    """
    Runs Inference using MCMC.
    """
    obs = jax.numpy.array(obs)

    nuts_kernel = numpyro.infer.NUTS(ai_model)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=500,
        num_chains=4,
        num_samples=5000)
    mcmc.run(jax.random.PRNGKey(int(time.time() * 1E6)), obs=jax.numpy.array(obs))
    mcmc.print_summary()
    return mcmc


def posterior(mcmc):
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.title("posterior")
        plt.xlabel("p")
        plt.hist(mcmc.get_samples()['p'], density=True, bins="auto", label="approximate")
        plt.show()


def posterior_predictive(obs, mcmc):
    """
    Evaluates Posterior Predictive from the Posterior Dist.
    In basic words, what we are most expecting to see. (Probability to see each observation given our posterior dist.)
    """
    posterior_predi = numpyro.infer.Predictive(ai_model, posterior_samples=mcmc.get_samples())
    posterior_samples = posterior_predi(jax.random.PRNGKey(int(time.time() * 1E6)),
                                        nobs={key: len(value) for (key, value) in obs})
    if utils.PLOT:
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.title("posterior predictive")
        for i in range(3):
            for j in range(3):
                ((func, action_param), action_obs) = obs[i % 3 + j % 3]
                key = f"o({str(func)}, {str(action_param)})"
                axs[i, j].set_title(f"Cell: {i * 3 + j + 1}")
                axs[i, j].set_xlim(-1, len(action_obs) + 1)
                axs[i, j].hist([sum(o) for o in posterior_samples[key]], density=True, bins=len(action_obs) * 2 + 1,
                               label="imaginations")
                axs[i, j].axvline(sum(action_obs), color="red", lw=2, label="observation")
                axs[i, j].legend()
        plt.show()
    return posterior_predi


# def p_value(obs, posterior_samples):
#     p_value = sum(posterior_samples['o'].sum(axis=1) >= sum(obs)) / len(posterior_samples['o'])
#     print(f"p_value = {p_value:.3f}")


def summarize_posterior(mcmc):
    """
    Summarizes Posterior, displays attributes such as mean, standard deviation, quantiles.
    """
    p = mcmc.get_samples()["p"]
    p_mean = p.mean()
    p_stddev = p.std()
    quantiles = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]
    pq = numpy.quantile(p, quantiles)
    print(f"stat\tp\n-------------")
    print(f"mean\t{p_mean:.3f}")
    print(f"stddev\t{p_stddev:.3f}")
    for i in range(len(quantiles)):
        print(f"{quantiles[i] * 100:3.0f}%\t{pq[i]:.3f}")
    if utils.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlabel("p")
        height, _, _ = plt.hist(p, histtype="step", lw=2, bins="auto", label="posterior")
        plt.title(f"mean={p_mean:.3f}, stddev={p_stddev:.3f}")
        plt.axvline(p_mean, ls="dashed", color="red", label="mean")
        plt.fill_betweenx([0, height.max()], pq[1], pq[-2],
                          color="red", alpha=0.1, label=f"{(quantiles[-2] - quantiles[1]) * 100:.0f}%")
        plt.fill_betweenx([0, height.max()], pq[2], pq[-3],
                          color="red", alpha=0.2, label=f"{(quantiles[-3] - quantiles[2]) * 100:.0f}%")
        plt.legend()
        plt.show()
    return p_mean, p_stddev


def ML(obs_file, prior_model_parameters=None):
    """
    Using the Logs in observations.log,
    """
    obs_map = parse_obs(obs_file)
    posterior_model_parameters = {}
    global model_beta_parameters
    prior_predi = prior_predictive(obs_map)
    mcmc = inference(obs_map)
    posterior(mcmc)
    posterior_predi = posterior_predictive(obs_map, mcmc)
    # p_value(obs, posterior_samples)
    p_mean, p_stddev = summarize_posterior(mcmc)
    # sample = numpy.random.choice(mcmc.get_samples()["p"])
    # print(f"sample: {sample}\n")
    # posterior_model_parameters[(func, action_parameter)] = sample
    # model_beta_parameters[(func, action_parameter)]['alpha'] = p_mean * (
    #         ((p_mean * (1 - p_mean)) / p_stddev) - 1)
    # model_beta_parameters[(func, action_parameter)]['beta'] = (1 - p_mean) * (
    #         ((p_mean * (1 - p_mean)) / p_stddev) - 1)
    fill_post_model_params(posterior_model_parameters, prior_model_parameters)
    print(f"posterior model parameters: {posterior_model_parameters}")
    return posterior_model_parameters
