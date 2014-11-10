#include "laplace.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(LaplaceTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Laplace<double> dist(0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  auto indexes = Distribution<double>::sort_data(samples);

  double likelihood1 = dist.log_likelihood(samples);

  dist.MLE(samples, indexes);

  double likelihood2 = dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood1);

  EXPECT_LT(0, dist.get_lambda());
}

TEST(LaplaceTest, Samples) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Laplace<double> dist(0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  for (size_t i = 0; i < n_samples; i++) {
    EXPECT_LT(-10, samples(i,0));
    EXPECT_GT(10, samples(i,0));
  }
}

TEST(LaplaceTest, ExtremeSamples) {
  Laplace<double> dist(0, 1);
  Array<double> samples({3,1});
  samples(0,0) = -1;
  samples(1,0) = 0;
  samples(2,0) = 1;

  double w1[] = {1, 1, 10}, w2[] = {10, 1, 1};
  auto indexes = Distribution<double>::sort_data(samples);

  {
    MA::ConstArray<double> weight({3}, w1);
    dist.MLE(samples, weight, indexes);
    EXPECT_LT(dist.get_mu(), samples(2,0));
    EXPECT_GT(dist.get_mu(), samples(1,0));
  }

  {
    MA::ConstArray<double> weight({3}, w2);
    dist.MLE(samples, weight, indexes);
    EXPECT_LT(dist.get_mu(), samples(1,0));
    EXPECT_GT(dist.get_mu(), samples(0,0));
  }
}
