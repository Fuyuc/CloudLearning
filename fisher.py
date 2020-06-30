import sys, glob, argparse
import numpy as np
import math, cv2
from cv2 import ml
from scipy.stats import multivariate_normal

def dictionary(descriptors, N):
	covs = [np.zeros((N, N), np.float64)]
	em = ml.EM_create()
	em.setClustersNumber(N)
	em.trainEM(descriptors)
	means = em.getMeans()
	covs = em.getCovs(covs)
	weights = em.getWeights()
	# print("各参数维数：")
	# print("均值："),print(np.float32(means).shape)
	# print("协方差："),print(np.float32(covs).shape)
	# print("权重："),print(np.float32(weights).shape)
	return np.float32(means), np.float32(covs), np.float32(weights)[0]

def likelihood_moment(x, ytk, moment):
	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
	return x_moment * ytk
#
def likelihood_statistics(samples, means, covs, weights):
	gaussians, s0, s1,s2 = {}, {}, {}, {}

	g = [multivariate_normal(mean=means[k], cov=covs[k],allow_singular=True) for k in range(0, len(weights)) ]
	for index, x in enumerate(samples):
		gaussians[index] = np.array([g_k.pdf(x) for g_k in g])   #存储x在mean[k]取值下的可能性

	for k in range(0, len(weights)):
		s0[k], s1[k], s2[k] = 0, 0, 0
		for index, x in enumerate(samples):
			probabilities = np.multiply(gaussians[index], weights)
			if ((probabilities == np.zeros((1,len(probabilities)),dtype = np.float64))).all() == False:
				probabilities = probabilities / np.sum(probabilities)
			s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
			s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
			s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

	return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
	return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, w):
	s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
	T = samples.shape[0]
	covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
	a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
	b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
	c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
	fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
	fv = normalize(fv)
	return fv

def generate_gmm(feat_vec, N):
	print("Training GMM of size", N)
	means, covs, weights = dictionary(feat_vec, N)

	th = 1 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

	return means, covs, weights