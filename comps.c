#include "comps.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_errno.h>

#define square(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))

double digamma(double x) {
  gsl_set_error_handler_off();
  gsl_sf_result result;
  if (gsl_sf_psi_e(x, &result)) {
    return NAN;
  }
  return result.val;
}

double trigamma(double x) {
  gsl_set_error_handler_off();
  gsl_sf_result result;
  if (gsl_sf_psi_1_e(x, &result)) {
    return NAN;
  }
  return result.val;
}

double lgamma(double x) {
  gsl_set_error_handler_off();
  gsl_sf_result result;
  if (gsl_sf_lngamma_e(x, &result)) {
    return NAN;
  }
  return result.val;
}

void print_matrix(double *mat, int m, int n) {
  printf("{");
  for (int i = 0; i < m; i++) {
    printf("{");
    for (int j = 0; j < n; j++) {
      printf("%f,", mat[i*n+j]);
    }
    printf("},");
  }
  printf("}\n");
}

double *malloc_vector(int len) {
  double *vector = (double *) calloc(len, sizeof(double));
  return vector;
}

double *malloc_matrix(int m, int n) {
  double *mat = (double *) calloc(m * n, sizeof(double));
  return mat;
}

double dot_product(double *A, double *B, int len) {
  double dot = 0;
  for (int i = 0; i < len; i++) {
    dot += A[i] * B[i];
  }
  return dot;
}

double *matrix_multiply(double *A, double *B, int n, int m, int p) {
  double *C = malloc_matrix(n, p);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) {
	C[i*p+j] += A[i*m+k] * B[k*p+j];
      }
    }
  }
  return C;
}

double sum(double *a, int length) {
  double sum = 0;
  for (int i = 0; i < length; i++) {
    sum += a[i];
  }
  return sum;
}

double gamma_covariance(double *gamma, int K, int i, int j) {
  double gamma_0 = sum(gamma, K);
  double denom = square(gamma_0) * (gamma_0 + 1);
  double cov;
  if (i == j) {
      cov = gamma[i] * (gamma_0 - gamma[i]) / denom;
  }
  else {
    cov = -gamma[i] * gamma[j] / denom;
  }
  return cov;
}

double *gamma_covariance_matrix(double *gamma, int K) {
  double *covs = malloc_matrix(K, K);
  for (int i = 0; i < K; i++) {
    for (int j = i; j < K; j++) {
      double elt = gamma_covariance(gamma, K, i, j);
      covs[i*K+j] = elt;
      covs[j*K+i] = elt;
    }
  }
  return covs;
}

double compute_M(double *gamma, double *eta, int K) {
  double *covs = gamma_covariance_matrix(gamma, K);
  double *temp = matrix_multiply(covs, eta, K, K, 1);
  double *prod = matrix_multiply(eta, temp, 1, K, 1);
  double M = prod[0];
  free(covs);
  free(temp);
  free(prod);
  return M;
}

double hessian_term(double *gamma, int K, int i, int a, int b) {
  double gamma_0 = sum(gamma, K);
  double term = square(gamma_0) * (gamma_0 + 1);
  double term_sq = term * term;
  double term2 = 3 * square(gamma_0) + 2 * gamma_0;
  double result;
  if (a == b) {
    double gamma_minus_a = sum(gamma, K) - gamma[a];
    if (i == a) {
      result = (gamma_minus_a * term - term2 * gamma[a] * gamma_minus_a) / term_sq;
    }
    else {
      result = (gamma[a] * term - term2 * gamma[a] * gamma_minus_a) / term_sq;
    }
  }
  else {
    if (i == a) {
      result = (-gamma[b] * term + term2 * gamma[a] * gamma[b]) / term_sq;
    }
    else if (i == b) {
      result = (-gamma[a] * term + term2 * gamma[a] * gamma[b]) / term_sq;
    }
    else {
      result = gamma[a] * gamma[b] * term2 / term_sq;
    }
  }
  return result;
}

double dM_dgamma_i(double *gamma, double *eta, int K, int i) {
  double *dcovs = malloc_matrix(K, K);
  for (int a = 0; a < K; a++) {
    for (int b = 0; b < K; b++) {
      dcovs[a*K+b] = hessian_term(gamma, K, i, a, b);
    }
  }
  
  double *temp = matrix_multiply(dcovs, eta, K, K, 1);
  double *prod = matrix_multiply(eta, temp, 1, K, 1);
  double deriv = prod[0];
  free(dcovs);
  free(temp);
  free(prod);
  return deriv;
}

double dM_deta_i(double *gamma, double *eta, int K, int i) {
  double *covs = gamma_covariance_matrix(gamma, K);
  double deriv = 0;
  for (int j = 0; j < K; j++) {
    deriv += eta[j] * (covs[i*K+j] + covs[j*K+i]);
  }
  free(covs);
  return deriv;
}

double likelihood_gamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int K, int y) {
  double gamma_0 = sum(gamma, K);
  double psi_gamma_0 = digamma(gamma_0);
  double lngamma_gamma_0 = lgamma(gamma_0);
  double Q = dot_product(gamma, eta, K) / gamma_0;
  double P = exp(Q);
  double M = compute_M(gamma, eta, K);
  double const_terms = -lngamma_gamma_0 + (y * Q - log(1 + P * (1 + M / 2))) / sigma;
  double sum = const_terms;
  for (int i = 0; i < K; i++) {
    double coef = digamma(gamma[i]) - psi_gamma_0;
    sum += coef * (alpha + phi_sum[i] - gamma[i]) + lgamma(gamma[i]);
  }
  return sum;
}

double compute_dgamma_i(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int K, int y, int i) {
  double gamma_0 = sum(gamma, K);
  double P = exp(dot_product(gamma, eta, K) / gamma_0);
  double term1 = trigamma(gamma[i]) * (alpha + phi_sum[i] - gamma[i]);
  double temp_sum = 0;
  for (int j = 0; j < K; j++) {
    temp_sum += alpha + phi_sum[j] - gamma[j];
  }
  double term2 = trigamma(gamma_0) * temp_sum;
  double M = compute_M(gamma, eta, K);
  
  double coef = (eta[i] * gamma_0 - dot_product(eta, gamma, K)) / square(gamma_0);
  double term3 = coef * y;
  double term4 = (P * coef * (1 + M / 2) + dM_dgamma_i(gamma, eta, K, i)) / (1 + P * (1 + M / 2));
  return term1 - term2 + (term3 - term4) / sigma;
}

double *compute_dgamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int K, int y) {
  double *dgammas = malloc_vector(K);
  for (int i = 0; i < K; i++) {
    dgammas[i] = compute_dgamma_i(alpha, gamma, phi_sum, eta, sigma, K, y, i);
  }
  return dgammas;
}

double likelihood_eta(double *gamma, double *eta, double sigma, int K, int y) {
  double gamma_0 = sum(gamma, K);
  double Q = dot_product(gamma, eta, K) / gamma_0;
  double P = exp(Q);
  double M = compute_M(gamma, eta, K);
  double term1 = y * Q;
  double term2 = log(1 + P * (1 + M / 2));
  return (term1 - term2) / sigma;
}

double *compute_deta(double *gamma, double *eta, double sigma, int K, int y) {
  double *detas = malloc_vector(K);
  double gamma_0 = sum(gamma, K);
  double P = exp(dot_product(gamma, eta, K) / gamma_0);
  double M = compute_M(gamma, eta, K);
  for (int i = 0; i < K; i++) {
    double coef = gamma[i] / gamma_0; 
    double term1 = y * coef;
    double term2 = (P * coef * (1 + M / 2) + dM_deta_i(gamma, eta, K, i)) / (1 + P * (1 + M / 2));
    detas[i] = (term1 - term2) / sigma;
  }
  return detas;
}

double likelihood_eta_batch(double *gammas, double *eta, double sigma, int *ys, int K, int N) {
  double sum = 0;
  for (int n = 0; n < N; n++) {
    sum += likelihood_eta(&gammas[n*K], eta, sigma, K, ys[n]);
  }
  return sum;
}

double *compute_deta_batch(double *gammas, double *eta, double sigma, int *ys, int K, int N) {
  double *detas_sum = (double *) calloc(K, sizeof(double));
  for (int n = 0; n < N; n++) {
    double *detas = compute_deta(&gammas[n*K], eta, sigma, K, ys[n]);
    for (int i = 0; i < K; i++) {
      detas_sum[i] += detas[i];
    }
    free(detas);
  }
  return detas_sum;
}

/* double compute_dsigma(double *gammas, double *eta, double sigma, int *ys, int K, int N) { */
/*   double ss = 0; */
/*   for (int n = 0; n < N; n++) { */
/*     double *gamma = &gammas[n*K]; */
/*     int y = ys[n]; */
/*     double gamma_0 = sum(gamma, K); */
/*     double Q = dot_product(gamma, eta, K) / gamma_0; */
/*     double P = exp(Q); */
/*     double M = compute_M(gamma, eta, K); */
/*     ss += y * Q - log(1 + P * (1 + M / 2)); */
/*   } */
/*   return -ss / square(sigma); */
/* } */

/* double optimize_sigma(double *gammas, double *eta, int *ys, int K, int N) { */
/*   double num = 0, denom = 0; */
/*   for (int n = 0; n < N; n++) { */
/*     double *gamma = &gammas[n*K]; */
/*     int y = ys[n]; */
/*     double gamma_0 = sum(gamma, K); */
/*     double Q = dot_product(gamma, eta, K) / gamma_0; */
/*     double P = exp(Q); */
/*     double M = compute_M(gamma, eta, K); */
/*     num += y * Q; */
/*     denom += log(1 + P * (1 + M / 2)); */
/*   } */
/*   return num / denom; */
/* } */
