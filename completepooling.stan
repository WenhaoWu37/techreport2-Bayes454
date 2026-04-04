// Technical Report 2 complete pooling strategy: Curling
data {
  int<lower=1> N; // number of rows (ends)
  vector<lower=0, upper=1>[N] x; // PowerPlay indicator 
  vector<lower=0, upper=4>[N] z; // points
  // to figure out how to type this I used https://mc-stan.org/docs/cmdstan-guide/example_model_data.html 
  // in stan, a vector can only be real numbers, so we need to use an array
  array[N] int<lower=0, upper=1> y; // USWin data (indicator variable)
  
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
  vector<lower=0, upper=1>[N] p;
  p = exp(beta0 + beta1*x + beta2*z);
}

model {
  target += bernoulli_lpmf(y | p);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(beta2 | m2, s2);
}
