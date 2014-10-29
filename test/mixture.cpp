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
  Normal<double> dist(0, 1);
  auto dist2 = make_mixture(dist, dist);
  Array<double> samples;
  dist2.sample(samples, n_samples, rng);
  for (size_t i = 0; i < n_samples; i++) {
    EXPECT_LT(-5, samples(i,0));
    EXPECT_GT(5, samples(i,0));
  }
}
