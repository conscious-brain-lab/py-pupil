#!/usr/bin/env python
# encoding: utf-8
"""
Random functions

Created by Stijn Nuiten on 2018-02-14.
Copyright (c) 2018 __MyCompanyName__. All rights reserved.
"""

def SDT(target, hit, fa):
	"""
	Calculate d' and criterion     """

	import numpy as np
	import scipy as sp
	import matplotlib.pyplot as plt
	import scipy.stats as stats 

	target = np.array(target, dtype=bool)
	hit = np.array(hit, dtype=bool)
	fa = np.array(fa, dtype=bool)

	hit_rate = (np.sum(hit)) / (float(np.sum(target)))
	fa_rate = (np.sum(fa)) / (float(np.sum(~target)))

	if hit_rate == 1:
		hit_rate = hit_rate-0.00001
	elif hit_rate == 0:
		hit_rate = hit_rate+0.00001

	if fa_rate == 1:
		fa_rate = fa_rate-0.00001
	elif fa_rate == 0:
		fa_rate = fa_rate+0.00001

	hit_rate_z = stats.norm.isf(1-hit_rate)
	fa_rate_z = stats.norm.isf(1-fa_rate)

	d = hit_rate_z - fa_rate_z
	c = -(hit_rate_z + fa_rate_z) / 2.0

	return(d, c)


def cluster_ttest(self, cond1, cond2, n_permutes, pval):
# % cluster_ttest runs cluster-base corrected pairwise t-tests for time-frequency data. 
# % This function has been adapted from scripts by Michael X Cohen.
# % Inputs are:
# %     - data for two conditions you want to test (subject x frequency x time)
# %     - number of permutations
# %     - threshold p-value
# % It returns the t-values for significant clusters (non-cluster t-values are set to 0).
			
	zval=sp.stats.norm.ppf(1-pval);
	# Now, let's start statistics stuff
	tnum = np.mean(cond2-cond1,axis=0);
	tdenom = np.std(np.array([cond1,cond2]).mean(axis=0),axis=0,ddof=1)**2/np.sqrt(cond1.shape[0]);  
	real_t = tnum/tdenom;

	# % initialize null hypothesis matrices
	permuted_tvals  = np.zeros((n_permutes,real_t.shape[0], real_t.shape[1]))
	max_cluster_sizes = np.zeros((1,n_permutes))
	# % generate pixel-specific null hypothesis parameter distributions
	for permi in range(n_permutes):
	    # % permuted condition mapping
		fake_condition_mapping = sign(np.random.normal(size=cond1.shape[0]))
		# % compute t-map of null hypothesis
		tnumfake = np.array([[(cond2[:,i,j]-cond1[:,i,j]) * fake_condition_mapping for i in range(cond1.shape[1])] for j in range(cond1.shape[2])]).transpose(2,1,0).mean(axis=0)
		# tnumfake = squeeze(mean(bsxfun(@times,cond2-cond1,fake_condition_mapping),1))
		permuted_tvals[permi,:,:] = tnumfake/tdenom;

	# % compute mean and standard deviation (used for thresholding)
	mean_h0 = np.mean(permuted_tvals,axis=0)
	var_h0  = np.std(permuted_tvals,axis=0, ddof=1);

	# % loop through permutations to cluster sizes under the null hypothesis
	for permi in range(n_permutes):
		threshimg = permuted_tvals[permi,:,:];
		threshimg = (threshimg-mean_h0)/var_h0; # Normalize (transform t- to Z-value)
		threshimg[abs(threshimg)<zval] = 0 # % threshold image at p-value

		# threshimg[abs(threshimg)<zval] = 0;
		# % find clusters (need image processing toolbox for this!)
		labeled, islands = measurements.label(threshimg)
		if islands>0:
			area = measurements.sum((threshimg!=0), labeled, index=arange(labeled.max() + 1))
			max_cluster_sizes[0,permi] = np.max(area)

	# % find clusters (need image processing toolbox for this!)
	cluster_thresh = np.percentile(max_cluster_sizes,100-(100*pval))
	print cluster_thresh
	# % now threshold real data...
	real_t_thresh = (real_t-mean_h0)/var_h0; # % first Z-score

	real_t_thresh[abs(real_t_thresh)<zval] = 0 	# % next threshold image at p-value

	# % now cluster-based testing
	real_island, realnumclust = measurements.label(real_t_thresh)
	for i in range(realnumclust):
	    # %if real clusters are too small, remove them by setting to zero!  
	    if sum(real_island==i+1)<cluster_thresh:
	        real_t_thresh[real_island==i]=0
	return np.array(real_t_thresh,dtype=bool)
	
