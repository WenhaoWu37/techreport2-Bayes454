// Technical Report 2 complete pooling strategy: Curling
data {
  int<lower=1> N; // number of rows (ends)
  real<lower=0, upper=1> y[N]; // USWin data (indicator variable)
  real<lower=0, upper=1> x[N]; // PowerPlay indicator 
  real<lower=0, upper=4> z[N]; // points
  
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
  real<lower=0, upper=1> p[N];
  p = beta0 + beta1*x + beta2*z; // error here?
}


model {
  target += bernoulli_lpmf(y | p);
  target += normal_lpdf(beta0 | m0, s0);
  target += normal_lpdf(beta1 | m1, s1);
  target += normal_lpdf(beta2 | m2, s2);
}



