// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; // number of rows (ends)
  vector[N] y; // USWin data
  real<lower=0, upper=1>[N] x; // PowerPlay
  real<lower-0, upper=4>[N] z;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real mu;
  real<lower=0> sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ normal(mu, sigma);
}

