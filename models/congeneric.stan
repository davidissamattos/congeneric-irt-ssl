// IRT Congeneric model
// Author:David Issa Mattos
// Date: 25 March 2021

data {
  int<lower=0> N; // size of the vector
  vector[N] y; // response of the item
  int p[N]; // test taker index(the model)
  int<lower=0> Np; // number of test takes (number of models)
  int item[N]; // item index of the test (the dataset)
  int<lower=0> Nitem; // number of items in the test
}


parameters {
 real<lower=0> b[Nitem]; // difficulty parameter
 real<lower=0> a[Nitem]; // discrimination parameter
 real<lower=0> theta[Np]; // ability of the test taker
 real<lower=0> sigma;
}

model {
  real mu[N];

  //Weakly informative priors
  b ~ normal(0, 1);
  a ~ normal(0,1);
  theta ~ normal(0,3);
  sigma ~ normal(0,1);//halfnormal

  //Linear gaussian model
  for(i in 1:N){
    mu[i] = b[item[i]] + a[item[i]]*theta[p[i]];
  }
  y ~ normal(mu, sigma);

}

generated quantities{
  vector[N] log_lik;
  vector[N] y_rep;
  for(i in 1:N){
    real mu;
    mu = b[item[i]] + a[item[i]]*theta[p[i]];
    log_lik[i] = normal_lpdf(y[i] | mu, sigma );
    y_rep[i] = normal_rng( mu, sigma);
  }
}
