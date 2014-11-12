#include "discrete.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(DiscreteTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Discrete<5, double> dist;
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);

  dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood1);

  auto p = dist.get_p();
  double sum = 0;
  for (size_t i = 0; i < p.size(); i++) {
    EXPECT_LE(0, p[i]);
    sum += p[i];
  }
  EXPECT_DOUBLE_EQ(1, sum);
}

TEST(DiscreteTest, MLE) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Discrete<5, double> dist;
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  dist.MLE(samples);

  std::vector<double> p = dist.get_p();
  double eps = 1e-2;
  double ll = dist.log_likelihood(samples);

  for (int i = 0; i < 5; i++) {
    p[i] += eps;
    dist.set_p(p);
    EXPECT_GE(ll, dist.log_likelihood(samples));
    p[i] -= 2* eps;
    dist.set_p(p);
    EXPECT_GE(ll, dist.log_likelihood(samples));
    p[i] += eps;
  }
}

TEST(DiscreteTest, Samples) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  {
    Discrete<2, double> dist;
    Array<double> samples;
    dist.sample(samples, n_samples, rng);
    size_t count_0 = 0, count_1 = 0;
    for (size_t i = 0; i < n_samples; i++) {
      if (samples(i,0) == 1)
        count_0++;
      if (samples(i,1) == 1)
        count_1++;
    }
    EXPECT_EQ(n_samples, count_0 + count_1);
    EXPECT_LT(0, count_0);
    EXPECT_LT(0, count_1);
  }
  {
    Discrete<2, double> dist({0,1});
    Array<double> samples;
    dist.sample(samples, n_samples, rng);
    size_t count_0 = 0, count_1 = 0;
    for (size_t i = 0; i < n_samples; i++) {
      if (samples(i,0) == 1)
        count_0++;
      if (samples(i,1) == 1)
        count_1++;
    }
    EXPECT_EQ(0, count_0);
    EXPECT_EQ(n_samples, count_1);
  }
}
