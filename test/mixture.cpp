#include "mixture.hpp"
#include "normal.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

/*TEST(NormalTest, Likelihood) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Normal<double> dist(0, 1);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);

  dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood1);

  EXPECT_LT(0, dist.get_sigma());
}*/

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
