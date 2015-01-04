#include "comps_categorical.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>

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

double compute_elognorm(double *gamma, double *eta, int C, int K) {
  double gamma_0 = sum(gamma, K);
  double elognorm = 0;
  for (int c = 0; c < C; c++) {
    double P_c = exp(dot_product(gamma, &eta[c*K], K) / gamma_0);
    elognorm += P_c;
  }
  elognorm = log(elognorm);
  return elognorm;
}

double likelihood_gamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int C, int K, int y) {
  double gamma_0 = sum(gamma, K);
  double psi_gamma_0 = digamma(gamma_0);
  double lngamma_gamma_0 = lgamma(gamma_0);
  double Q = dot_product(gamma, &eta[y*K], K) / gamma_0;
  double elognorm = compute_elognorm(gamma, eta, C, K);
  double const_terms = -lngamma_gamma_0 + (Q - elognorm) / sigma;
  double sum = const_terms;
  for (int i = 0; i < K; i++) {
    double coef = digamma(gamma[i]) - psi_gamma_0;
    sum += coef * (alpha + phi_sum[i] - gamma[i]) + lgamma(gamma[i]);
  }
  return sum;
}

double compute_dgamma_i(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int C, int K, int y, int i) {
  double gamma_0 = sum(gamma, K);
  double term1 = trigamma(gamma[i]) * (alpha + phi_sum[i] - gamma[i]);
  double temp_sum = 0;
  for (int j = 0; j < K; j++) {
    temp_sum += alpha + phi_sum[j] - gamma[j];
  }
  double term2 = trigamma(gamma_0) * temp_sum;
  double term3 = (eta[y*K+i] * gamma_0 - dot_product(&eta[y*K], gamma, K)) / square(gamma_0);
  double term4_denom = 0;
  double term4_num = 0;
  for (int c = 0; c < C; c++) {
    double P_c = exp(dot_product(gamma, &eta[c*K], K) / gamma_0);
    double coef = (eta[c*K+i] * gamma_0 - dot_product(&eta[c*K], gamma, K)) / square(gamma_0);
    term4_num += P_c * coef;
    term4_denom += P_c;
  }
  double term4 = term4_num / term4_denom;
  return term1 - term2 + (term3 - term4) / sigma;
}

double *compute_dgamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int C, int K, int y) {
  double *dgammas = malloc_vector(K);
  for (int i = 0; i < K; i++) {
    dgammas[i] = compute_dgamma_i(alpha, gamma, phi_sum, eta, sigma, C, K, y, i);
  }
  return dgammas;
}

double likelihood_eta(double *gamma, double *eta, double sigma, int C, int K, int y) {
  double gamma_0 = sum(gamma, K);
  double term1 = dot_product(gamma, &eta[y*K], K) / gamma_0;
  double term2 = compute_elognorm(gamma, eta, C, K);
  return (term1 - term2) / sigma;
}

double *compute_deta(double *gamma, double *eta, double sigma, int C, int K, int y) {
  double *detas = malloc_matrix(C, K);
  double gamma_0 = sum(gamma, K);
  double term2_denom = 0;
  for (int j = 0; j < C; j++) {
    double P_c = exp(dot_product(gamma, &eta[j*K], K) / gamma_0);
    term2_denom += P_c;
  }
  for (int c = 0; c < C; c++) {
    double P_c = exp(dot_product(gamma, &eta[c*K], K) / gamma_0);
    for (int i = 0; i < K; i++) {
      double term1 = gamma[i] / gamma_0;
      double term2_num = P_c * term1;
      double term2 = term2_num / term2_denom;
      if (c == y) {
	detas[c*K+i] = (term1 - term2) / sigma;
      }
      else {
	detas[c*K+i] = -term2 / sigma;
      }
    }
  }
  return detas;
}

double likelihood_eta_batch(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N) {
  double sum = 0;
  for (int n = 0; n < N; n++) {
    sum += likelihood_eta(&gammas[n*K], eta, sigma, C, K, ys[n]);
  }
  return sum;
}

double *compute_deta_batch(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N) {
  double *detas_sum = (double *) calloc(C * K, sizeof(double));
  for (int n = 0; n < N; n++) {
    double *detas = compute_deta(&gammas[n*K], eta, sigma, C, K, ys[n]);
    for (int j = 0; j < C * K; j++) {
      detas_sum[j] += detas[j];
    }
    free(detas);
  }
  return detas_sum;
}

/* double compute_dsigma(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N) { */
/*   double ss = 0; */
/*   for (int n = 0; n < N; n++) { */
/*     double *gamma = &gammas[n*K]; */
/*     int y = ys[n]; */
/*     double gamma_0 = sum(gamma, K); */
/*     double Q = dot_product(gamma, &eta[y*K], K) / gamma_0; */
/*     double elognorm = compute_elognorm(gamma, eta, C, K); */
/*     ss += Q - elognorm; */
/*   } */
/*   return -ss / square(sigma); */
/* } */

/* double optimize_sigma(double *gammas, double *eta, int *ys, int C, int K, int N) { */
/*   double num = 0, denom = 0; */
/*   for (int n = 0; n < N; n++) { */
/*     double *gamma = &gammas[n*K]; */
/*     int y = ys[n]; */
/*     double gamma_0 = sum(gamma, K); */
/*     double Q = dot_product(gamma, &eta[y*K], K) / gamma_0; */
/*     double elognorm = compute_elognorm(gamma, eta, C, K); */
/*     num += Q; */
/*     denom += elognorm; */
/*   } */
/*   return num / denom; */
/* } */

