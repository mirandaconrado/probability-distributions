#include "asymmetric_normal.hpp"
#include "normal.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(AsymmetricNormalTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricNormal<double> dist1(0.25, 0, 1);
  Normal<double> dist2(0, 1);
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

  EXPECT_LT(0, dist1.get_sigma());
  EXPECT_LT(0, dist1.get_p());
  EXPECT_GT(1, dist1.get_p());
}

TEST(AsymmetricNormalTest, LikelihoodConsistency) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricNormal<double> dist1(0.5, 0, 1);
  dist1.fix_p(true);
  Normal<double> dist2(0, 1);
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

  EXPECT_NEAR(dist1.get_mu(), dist2.get_mu(), 1e-8);
  EXPECT_DOUBLE_EQ(dist1.get_sigma(), dist2.get_sigma());
}

TEST(AsymmetricNormalTest, MLE) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricNormal<double> dist(0.5, 0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  auto indexes = Distribution<double>::sort_data(samples);
  dist.MLE(samples, indexes);

  double p = dist.get_p(), mu = dist.get_mu(), sigma = dist.get_sigma();
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

  dist.set_sigma(sigma + eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_sigma(sigma - eps);
  EXPECT_GE(ll, dist.log_likelihood(samples));
  dist.set_sigma(sigma);
}

TEST(AsymmetricNormalTest, Samples) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricNormal<double> dist(0.5, 0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  for (size_t i = 0; i < n_samples; i++) {
    EXPECT_LT(-5, samples(i,0));
    EXPECT_GT(5, samples(i,0));
  }
}
