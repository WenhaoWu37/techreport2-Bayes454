// Technical Report 2 partial pooling strategy: Curling
data {
  int<lower=1> N; // number of rows
  vector<lower=0, upper=1>[N] x; // PowerPlay indicator 
  vector<lower=0, upper=4>[N] z; // points
  // to figure out how to type this I used https://mc-stan.org/docs/cmdstan-guide/example_model_data.html 
  // in stan, a vector can only be real numbers, so we need to use an array
  array[N] int<lower=0, upper=1> y; // USWin data (indicator variable)
  
  int<lower=0> n_competitions; // number of competitions
  int<lower=0> n_matches; // number of matches (total, all competitions)
  
  // indices
  int<lower=0> j[N]; // match
  int<lower=0> k[N]; // competition
  
  // hyper-hyper parameters
  real w0;
  real w1;
  real w2;
  real<lower=0> sigma0;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0> c0;
  real<lower=0> c1;
  real<lower=0> c2;
  real<lower=0> d0;
  real<lower=0> d1;
  real<lower=0> d2;
  real<lower=0> f0;
  real<lower=0> f1;
  real<lower=0> f2;
  real<lower=0> g0;
  real<lower=0> g1;
  real<lower=0> g2;
  real<lower=0> q0;
  real<lower=0> q1;
  real<lower=0> q2;
  real<lower=0> r0;
  real<lower=0> r1;
  real<lower=0> r2;
  
}

// The parameters accepted by the model
parameters {
  // prior layer: different regression coefficients for each match
  vector[n_matches] beta0;
  vector[n_matches] beta1;
  vector[n_matches] beta2;
  
  // hyper prior layer: different values for each competition
  vector[n_competitions] m0;
  vector[n_competitions] m1;
  vector[n_competitions] m2;
  vector<lower=0>[n_competitions] s0;
  vector<lower=0>[n_competitions] s1;
  vector<lower=0>[n_competitions] s2;
  
  // hyper-hyper prior layer
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
  vector<lower=0>[n_competitions] s02_inv;
  vector<lower=0>[n_competitions] s12_inv;
  vector<lower=0>[n_competitions] s22_inv;
  s02_inv = 1/(s0^2);
  s12_inv = 1/(s1^2);
  s22_inv = 1/(s2^2);
  /// hyper-hyper priors
  real<lower=0> v02_inv;
  real<lower=0> v12_inv;
  real<lower=0> v22_inv;
  v02_inv = 1/(v0^2);
  v12_inv = 1/(v1^2);
  v22_inv = 1/(v2^2);
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
  //hyper-hyper priors
  target += normal_lpdf(u0 | w0, sigma0);
  target += normal_lpdf(u1 | w1, sigma1);
  target += normal_lpdf(u2 | w2, sigma2);
  target += gamma_lpdf(v02_inv | c0, d0);
  target += gamma_lpdf(v12_inv | c1, d1);
  target += gamma_lpdf(v22_inv | c2, d2);
  target += gamma_lpdf(alpha0 | f0, g0);
  target += gamma_lpdf(alpha1 | f1, g1);
  target += gamma_lpdf(alpha2 | f2, g2);
  target += gamma_lpdf(eta0 | q0, r0);
  target += gamma_lpdf(eta1 | q1, r1);
  target += gamma_lpdf(eta2 | q2, r2);
}
