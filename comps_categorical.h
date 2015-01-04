double likelihood_eta_batch(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N);

double *compute_deta_batch(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N);

double *compute_dgamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int C, int K, int y);

double likelihood_gamma(double alpha, double *gamma, double *phi_sum, double *eta, double sigma, int C, int K, int y);

//double optimize_sigma(double *gammas, double *eta, int *ys, int C, int K, int N);

//double compute_dsigma(double *gammas, double *eta, double sigma, int *ys, int C, int K, int N);
