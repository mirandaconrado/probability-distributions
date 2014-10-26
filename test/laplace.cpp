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

  EXPECT_LT(0, dist.get_b());
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
