// Technical Report 2 complete pooling strategy: Curling

data {
  int<lower=0> N; // number of rows (ends)
  vector[N] y; // USWin data
  real<lower=0, upper=1>[N] x; // PowerPlay
  real<lower=0, upper=4>[N] z; // points
  
  // hyper parameters
  real m0;
  real m1;
  real m2;
  real<lower=0> s0;
  real<lower=0> s1;
  real<lower=0> s2;
  
}

// The parameters accepted by the model
parameters {
  real beta0;
  real beta1;
  real beta2;
}

transformed parameters {
  vector[N] p;
  p = beta0 + beta1 * x + beta2 * z;
}


model {
  target += normal_lpdf(y | p;
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += gamma_lpdf(beta2 | m2, s2);
}



