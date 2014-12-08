double likelihood_eta_batch(double *gammas, double *eta, int *ys, int C, int K, int N);

double *compute_deta_batch(double *gammas, double *eta, int *ys, int C, int K, int N);

double *compute_dgamma(double alpha, double *gamma, double *phi_sum, double *eta, int C, int K, int y);

double likelihood_gamma(double alpha, double *gamma, double *phi_sum, double *eta, int C, int K, int y);
