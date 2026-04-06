// Technical Report 2 partial pooling strategy: Curling
// only groups are the competitions
data {
  int<lower=1> N; // number of rows
  vector<lower=0, upper=1>[N] x; // PowerPlay indicator 
  vector<lower=0, upper=4>[N] z; // points
  // to figure out how to type this I used https://mc-stan.org/docs/cmdstan-guide/example_model_data.html 
  // in stan, a vector can only be real numbers, so we need to use an array
  array[N] int<lower=0, upper=1> y; // USWin data (indicator variable)
  
  int<lower=0> n_competitions; // number of competitions
  
  // indices
  int<lower=0> j[N]; // competition
  
  // hyperprior parameters
  real u0;
  real u1;
  real u2;
  real<lower=0> v0;
  real<lower=0> v1;
  real<lower=0> v2;
  real<lower=0> alpha0;
  real<lower=0> alpha1;
  real<lower=0> alpha2;
  real<lower=0> eta0;
  real<lower=0> eta1;
  real<lower=0> eta2;

}

// The parameters accepted by the model
parameters {
  // prior layer: different regression coefficients for each competition
  vector[n_competitions] beta0;
  vector[n_competitions] beta1;
  vector[n_competitions] beta2;
  
  // hyperprior layer
  real m0;
  real m1;
  real m2;
  real<lower=0> s0;
  real<lower=0> s1;
  real<lower=0> s2;
}

transformed parameters {
  // logistic regression in order to confine p to be between 0 and 1
  vector<lower=0>[N] odds;
  vector<lower=0, upper=1>[N] p;
  
  // p for each row based on match 
  for(l in 1:N){
    odds[l] = exp(beta0[j[l]] + beta1[j[l]]*x[l] + beta2[j[l]]*z[l]);
    p[l] = odds[l] ./ (odds[l] + 1);
  }
  
  // transform variance parameters (for Gamma distributions)
  /// hyperpriors
  real<lower=0> s02_inv;
  real<lower=0> s12_inv;
  real<lower=0> s22_inv;
  s02_inv = 1/(s0^2);
  s12_inv = 1/(s1^2);
  s22_inv = 1/(s2^2);
}

model {
  target += bernoulli_lpmf(y | p);
  // priors
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(beta2 | m2, s2);
  // hyperpriors
  target += normal_lpdf(m0 | u0, v0);
  target += normal_lpdf(m1 | u1, v1);
  target += normal_lpdf(m2 | u2, v2);
  target += gamma_lpdf(s02_inv | alpha0, eta0);
  target += gamma_lpdf(s12_inv | alpha1, eta1);
  target += gamma_lpdf(s22_inv | alpha2, eta2);
}
