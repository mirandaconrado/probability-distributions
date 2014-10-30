#include "mixture.hpp"
#include "normal.hpp"
#include "laplace.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(MixtureTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Normal<double> dist1(-10, 1);
  Normal<double> dist2(10, 1);
  auto dist = make_mixture(dist1, dist2);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);
  (void)likelihood1;

  dist.get_component<0>().set_mean(-0.1);
  dist.get_component<1>().set_mean(0.1);

  auto new_dist = make_mixture(Laplace<double>(-0.1, 1),
      Laplace<double>(0.1, 1));

  dist.MLE(samples);
  new_dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);
  double new_likelihood = new_dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood1);
  EXPECT_GE(likelihood2, new_likelihood);
}

TEST(MixtureTest, Samples) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Normal<double> dist1(-10, 1);
  Normal<double> dist2(10, 1);
  auto dist = make_mixture(dist1, dist2);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);
  unsigned int count_0 = 0, count_1 = 0;

  for (size_t i = 0; i < n_samples; i++) {
    if (samples(i,0) < 0) {
      EXPECT_LT(-15, samples(i,0));
      EXPECT_GT(-5, samples(i,0));
      count_0++;
    }
    else {
      EXPECT_LT(5, samples(i,0));
      EXPECT_GT(15, samples(i,0));
      count_1++;
    }
  }

  EXPECT_LT(0, count_0);
  EXPECT_LT(0, count_1);
}
