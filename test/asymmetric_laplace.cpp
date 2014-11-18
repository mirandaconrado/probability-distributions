#include "asymmetric_laplace.hpp"
#include "laplace.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(AsymmetricLaplaceTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricLaplace<double> dist1(0.25, 0, 1);
  Laplace<double> dist2(0, 1);
  Array<double> samples;
  dist1.sample(samples, n_samples, rng);

  auto indexes = Distribution<double>::sort_data(samples);

  double likelihood11 = dist1.log_likelihood(samples);
  double likelihood12 = dist2.log_likelihood(samples);

  EXPECT_GE(likelihood11, likelihood12);

  dist1.MLE(samples, indexes);
  dist2.MLE(samples, indexes);

  double likelihood21 = dist1.log_likelihood(samples);
  double likelihood22 = dist2.log_likelihood(samples);

  EXPECT_GE(likelihood22, likelihood12);
  EXPECT_GE(likelihood21, likelihood11);
  EXPECT_GT(likelihood21, likelihood22);

  EXPECT_LT(0, dist1.get_lambda());
  EXPECT_LT(0, dist1.get_p());
  EXPECT_GT(1, dist1.get_p());
}

TEST(AsymmetricLaplaceTest, LikelihoodConsistency) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricLaplace<double> dist1(0.5, 0, 1);
  dist1.fix_p(true);
  Laplace<double> dist2(0, 1);
  Array<double> samples;
  dist1.sample(samples, n_samples, rng);

  auto indexes = Distribution<double>::sort_data(samples);

  double likelihood11 = dist1.log_likelihood(samples);
  double likelihood12 = dist2.log_likelihood(samples);

  EXPECT_DOUBLE_EQ(likelihood12, likelihood11);

  dist1.MLE(samples, indexes);
  dist2.MLE(samples, indexes);

  double likelihood21 = dist1.log_likelihood(samples);
  double likelihood22 = dist2.log_likelihood(samples);

  EXPECT_DOUBLE_EQ(likelihood22, likelihood21);

  EXPECT_DOUBLE_EQ(dist1.get_mu(), dist2.get_mu());
  EXPECT_DOUBLE_EQ(dist1.get_lambda(), dist2.get_lambda());
}

TEST(AsymmetricLaplaceTest, MLE) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricLaplace<double> dist(0.5, 0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  auto indexes = Distribution<double>::sort_data(samples);
  dist.MLE(samples, indexes);

  double p = dist.get_p(), mu = dist.get_mu(), lambda = dist.get_lambda();
  // eps has to be higher because there's a continuum of minima
  double eps = 1e-2;
  double ll = dist.log_likelihood(samples);

  dist.set_p(p + eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_p(p - eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_p(p);

  dist.set_mu(mu + eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_mu(mu - eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_mu(mu);

  dist.set_lambda(lambda + eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_lambda(lambda - eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_lambda(lambda);
}

TEST(AsymmetricLaplaceTest, Samples) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricLaplace<double> dist(0.5, 0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  for (size_t i = 0; i < n_samples; i++) {
    EXPECT_LT(-10, samples(i,0));
    EXPECT_GT(10, samples(i,0));
  }
}
