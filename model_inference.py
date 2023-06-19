import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import jax
import numpy
import scipy
import time

import runtime

model_beta_parameters = {1: {'alpha': 1, 'beta': 1}, 2: {'alpha': 1, 'beta': 1}, 3: {'alpha': 1, 'beta': 1},
                         4: {'alpha': 1, 'beta': 1}, 5: {'alpha': 1, 'beta': 1}, 6: {'alpha': 1, 'beta': 1},
                         7: {'alpha': 1, 'beta': 1}, 8: {'alpha': 1, 'beta': 1}, 9: {'alpha': 1, 'beta': 1}}


# TODO: Change the model_parameters value to probability dist.
# TODO : The values of model_parameters need to be the value of the dist (for example [alpha,beta] for Beta Dist.)

# When inferring the AI Agent, and before sending the model_parameters, we need to sample
# from the dist and send those sampled values to the model_parameters of the AI Agent.
# What we are expecting to see is an increase in the acc. rewards.
# Order is -> 1. Run Model, 2. Gather logs / acc. reward, 3. Update Posterior (model_parameters) and repeat 1.,2.,3...

def ai_model(alpha, beta, obs=None, nobs=runtime.INIT_OBSERVATIONS_LEN):
    """
    (Tic-Tac-Toe) AI Agent model with NumPyro.
    """
    if obs is not None:  # for prior predictive
        nobs = len(obs)
    # p ~ Beta(1, 1)
    p = numpyro.sample("p", dist.Beta(alpha, beta))
    with numpyro.plate("obs", len(obs) if obs is not None else nobs):
        # o ~ Bernoulli(p)
        numpyro.sample("o", dist.Bernoulli(p), obs=obs)


def prior_predictive(obs, alpha, beta):
    """
    Evaluates Prior Predictive from the Prior Dist.
    Expecting to see mass in the bar where observation and imaginations are aligned.
    """
    prior_predi = numpyro.infer.Predictive(ai_model, num_samples=10000)
    prior_samples = prior_predi(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta)
    if runtime.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlim(-1, len(obs) + 1)
        plt.hist([sum(o) for o in prior_samples['o']], density=True, bins=len(obs) * 2 + 1,
                 label="imaginations")
        plt.axvline(sum(obs), color="red", lw=2, label="observation")
        plt.title("prior predictive")
        plt.legend()
        plt.show()
    return prior_predi


def inference(obs, alpha, beta):
    """
    Runs Inference using MCMC.
    """
    nuts_kernel = numpyro.infer.NUTS(ai_model)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=500,
        num_chains=4,
        num_samples=5000)
    mcmc.run(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta, obs=jax.numpy.array(obs))
    mcmc.print_summary()
    return mcmc


def posterior(mcmc):
    if runtime.PLOT:
        plt.figure(figsize=(10, 3))
        plt.title("posterior")
        plt.xlabel("p")
        plt.hist(mcmc.get_samples()['p'], density=True, bins="auto", label="approximate")
        plt.show()


def posterior_predictive(obs, mcmc, alpha, beta):
    """
    Evaluates Posterior Predictive from the Posterior Dist.
    In basic words, what we are most expecting to see. (Probability to see each observation given our posterior dist.)
    """
    posterior_predi = numpyro.infer.Predictive(ai_model, posterior_samples=mcmc.get_samples())
    posterior_samples = posterior_predi(jax.random.PRNGKey(int(time.time() * 1E6)), alpha=alpha, beta=beta)
    if runtime.PLOT:
        plt.figure(figsize=(10, 3))
        plt.xlim(-1, len(obs) + 1)
        plt.hist([sum(o) for o in posterior_samples['o']], density=True, bins=len(obs) * 2 + 1,
                 label="imaginations")
        plt.axvline(sum(obs), color="red", lw=2, label="observation")
        plt.title("posterior predictive")
        plt.legend()
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
    if runtime.PLOT:
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
    obs = {}
    with open(obs_file, "r+") as obs_log:
        for line in obs_log.readlines():
            line = line.replace(" ", "").strip("\n").split(":")
            cell = int(line[0])
            obs[cell] = eval(line[1])
    posterior_model_parameters = {}
    global model_beta_parameters
    for cell, prob in prior_model_parameters.items():
        print(f"Cell: {cell}")
        alpha = model_beta_parameters[cell]['alpha']
        beta = model_beta_parameters[cell]['beta']
        print(f"alpha: {alpha}, beta: {beta}")
        prior_predi = prior_predictive(obs[cell], alpha, beta)
        mcmc = inference(obs[cell], alpha, beta)
        posterior(mcmc)
        posterior_predi = posterior_predictive(obs[cell], mcmc, alpha, beta)
        # p_value(obs, posterior_samples)
        p_mean, p_stddev = summarize_posterior(mcmc)
        sample = numpy.random.choice(mcmc.get_samples()["p"])
        print(f"sample: {sample}\n")
        posterior_model_parameters[cell] = sample
        model_beta_parameters[cell]['alpha'] = p_mean * (((p_mean * (1 - p_mean)) / p_stddev) - 1)
        model_beta_parameters[cell]['beta'] = (1 - p_mean) * (((p_mean * (1 - p_mean)) / p_stddev) - 1)
    print(f"posterior model parameters: {posterior_model_parameters}")
    return posterior_model_parameters
