# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re, string, time
import numpy as n
cimport numpy as n
from numpy.linalg import norm
from scipy.special import gammaln, psi, polygamma
from scipy.optimize import minimize, line_search

import ctypes
from libc.stdlib cimport free

DTYPE = n.double
ctypedef n.double_t DTYPE_t

cdef extern from "comps.h":
    double likelihood_eta_batch(double *gammas, double *eta, double sigma, int *ys, int K, int N);

    double *compute_deta_batch(double *gammas, double *eta, double sigma, int *ys, int K, int N);

    double *compute_dgamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int K, int y);

    double likelihood_gamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int K, int y);

#    double optimize_sigma(double *gammas, double *eta, int *ys, int K, int N);

#    double compute_dsigma(double *gammas, double *eta, double sigma, int *ys, int K, int N);

n.random.seed(100000001)
meanchangethresh = 0.001

def double_array(n.ndarray[DTYPE_t] x):
    cdef n.ndarray[DTYPE_t] result = n.ascontiguousarray(x, dtype=n.double)
    return result

def int_array(n.ndarray[n.long_t] x):
    cdef n.ndarray[int] result = n.ascontiguousarray(x, dtype=ctypes.c_int)
    return result

def double_matrix(n.ndarray[DTYPE_t, ndim=2] x):
    cdef n.ndarray[DTYPE_t, ndim=2] result = n.ascontiguousarray(x, dtype=n.double)
    return result

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, zeta, tau0=64, kappa=0.7, sigma=1.):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._zeta = zeta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._eta = n.random.randn(self._K)
        self._sigma = sigma

    def _dgamma(self, gamma, phi_sum, y):
        cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(self._eta)
        cdef n.ndarray[DTYPE_t, mode="c"] phi_sum_c = double_array(phi_sum)
        cdef n.ndarray[DTYPE_t, mode="c"] gamma_c = double_array(gamma)
        cdef double *dgammas_c = compute_dgamma(self._alpha, &gamma_c[0], &phi_sum_c[0],
                                                  &eta_c[0], self._sigma, self._K, y);
        dgammas = n.zeros(self._K)
        for i in range(self._K):
            dgammas[i] = dgammas_c[i]
        free(dgammas_c)
        return dgammas

    def _likelihood_gamma(self, gamma, phi_sum, y):
        cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(self._eta)
        cdef n.ndarray[DTYPE_t, mode="c"] phi_sum_c = double_array(phi_sum)
        cdef n.ndarray[DTYPE_t, mode="c"] gamma_c = double_array(gamma)
        cdef double likelihood = likelihood_gamma(self._alpha, &gamma_c[0], &phi_sum_c[0], &eta_c[0], self._sigma, self._K, y)
        return likelihood

    def _deta(self, eta, gammas, ys):
        cdef n.ndarray[DTYPE_t, ndim=2, mode="c"] gamma_c = double_matrix(gammas)
        cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(eta)
        cdef n.ndarray[int, mode="c"] ys_c = int_array(ys)
        cdef double *detas_c = compute_deta_batch(&gamma_c[0,0], &eta_c[0], self._sigma, &ys_c[0], self._K, len(ys))

        detas = n.zeros(self._K)
        for i in range(self._K):
            detas[i] = detas_c[i]            
        free(detas_c)
        return detas

    def _likelihood_eta(self, eta, gammas, ys):
        cdef n.ndarray[DTYPE_t, ndim=2, mode="c"] gamma_c = double_matrix(gammas)
        cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(eta)
        cdef n.ndarray[int, mode="c"] ys_c = int_array(ys)
        cdef double likelihood = likelihood_eta_batch(&gamma_c[0,0], &eta_c[0], self._sigma, &ys_c[0], self._K, len(ys))
        return likelihood

    def _optimize_gamma(self, gamma, phi_sum, y, maxiter=20):
        f = lambda x: -self._likelihood_gamma(x, phi_sum, y)
        g = lambda x: -self._dgamma(x, phi_sum, y)
        new_gamma = gamma
        bounds = [(0, None) for i in range(len(gamma))]
        options = { "maxiter": maxiter }
        gamma = 1*n.random.gamma(100., 1./100., self._K)
        res = minimize(f, gamma, method='L-BFGS-B', jac=g, bounds=bounds, options=options)
        if res.success:
            new_gamma = res.x
        return new_gamma

    def _optimize_eta(self, eta, gammas, ys, maxiter=20):
        f = lambda x: -self._likelihood_eta(x, gammas, ys)
        g = lambda x: -self._deta(x, gammas, ys)
        new_eta = eta
        options = { "maxiter": maxiter }
        eta = n.random.randn(self._K)
        res = minimize(f, eta, method='L-BFGS-B', jac=g, options=options)
        if res.success:
            new_eta = res.x
        return new_eta

    # def _dsigma(self, gammas, ys):
    #     cdef n.ndarray[DTYPE_t, ndim=2, mode="c"] gamma_c = double_matrix(gammas)
    #     cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(self._eta)
    #     cdef n.ndarray[int, mode="c"] ys_c = int_array(ys)
    #     cdef double dsigma = compute_dsigma(&gamma_c[0,0], &eta_c[0], self._sigma, &ys_c[0], self._K, len(ys))
    #     return dsigma

    # def _optimize_sigma(self, gammas, ys):
    #     cdef n.ndarray[DTYPE_t, ndim=2, mode="c"] gamma_c = double_matrix(gammas)
    #     cdef n.ndarray[DTYPE_t, mode="c"] eta_c = double_array(self._eta)
    #     cdef n.ndarray[int, mode="c"] ys_c = int_array(ys)
    #     cdef double sigma = optimize_sigma(&gamma_c[0,0], &eta_c[0], &ys_c[0], self._K, len(ys))
    #     return sigma

    def estimate_gamma(self, docs):
        wordids, wordcts = parse_doc_list(docs, self._vocab)
        gamma, _ = self.do_e_step(wordids, wordcts)
        return gamma

    def do_e_step(self, wordids, wordcts, ys=None):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        Elogbeta = dirichlet_expectation(self._lambda)
        expElogbeta = n.exp(Elogbeta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                phi_sum = expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                if ys is None:
                    gammad = self._alpha + phi_sum
                else:
                    y = ys[d]
                    gammad = self._optimize_gamma(gammad, phi_sum, y)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * expElogbeta

        return((gamma, sstats))

    def update_lambda_all(self, docs, ys, max_iter=20):
        wordids, wordcts = parse_doc_list(docs, self._vocab)
        for i in range(max_iter):
            (gammas, sstats) = self.do_e_step(wordids, wordcts, ys)
            self._lambda = self._zeta + sstats
            self._eta = self._optimize_eta(self._eta, gammas, ys)
#            self._sigma = self._optimize_sigma(gammas, ys)
            print self._eta

    def update_lambda(self, wordids, wordcts, ys):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gammas, sstats) = self.do_e_step(wordids, wordcts, ys)
        # Estimate held-out likelihood for current values of lambda.

        # Update lambda based on documents.
        batchD = len(ys)
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._zeta + self._D * sstats / batchD)
        self._eta += rhot * self._deta(self._eta, gammas, ys) * self._D / batchD
#        self._sigma = self._sigma * (1-rhot) + \
 #         rhot * self._optimize_sigma(gammas, ys) * self._D / batchD
        
        self._updatect += 1
        print self._eta
        return gammas

    def train(self, docs, ys, batchsize):
        rest_docs = docs
        rest_ys = ys
        while len(rest_docs) > 0:
            batch_docs = rest_docs[:batchsize]
            batch_ys = rest_ys[:batchsize]
            rest_docs = rest_docs[batchsize:]
            rest_ys = rest_ys[batchsize:]
            wordids, wordcts = parse_doc_list(batch_docs, self._vocab)
            self.update_lambda(wordids, wordcts, batch_ys)

    def _predict(self, gamma):
        gamma_0 = gamma.sum()
        prod = self._eta.dot(gamma) / gamma_0
        if prod > 0:
            return 1
        else:
            return 0

    def predict(self, docs):
        wordids, wordcts = parse_doc_list(docs, self._vocab)
        gammas, _ = self.do_e_step(wordids, wordcts)
        preds = [self._predict(gamma) for gamma in gammas]
        return preds

    def approx_bound(self, wordids, wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        Elogbeta = dirichlet_expectation(self._lambda)
        expElogbeta = n.exp(Elogbeta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / batchD

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._zeta-self._lambda)*Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._zeta))
        score = score + n.sum(gammaln(self._zeta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)
