#include "asymmetric_laplace.hpp"
#include "asymmetric_normal.hpp"
#include "mixture.hpp"
#include "normal.hpp"
#include "laplace.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

using namespace MultidimensionalArray;
using namespace ProbabilityDistributions;

TEST(MixtureTest, LikelihoodNormal) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Normal<double> dist1(-10, 1);
  Normal<double> dist2(10, 1);
  AsymmetricNormal<double> dist3(0.5, -10, 1);
  AsymmetricNormal<double> dist4(0.5, 10, 1);
  dist3.fix_p(true);
  dist4.fix_p(true);
  auto dist = make_mixture(dist1, dist2);
  auto dist_a = make_mixture(dist3, dist4);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);
  double likelihood1_a = dist_a.log_likelihood(samples);
  EXPECT_DOUBLE_EQ(likelihood1, likelihood1_a);

  dist.get_component<0>().set_mean(-0.1);
  dist.get_component<1>().set_mean(0.1);
  dist_a.get_component<0>().set_mu(-0.1);
  dist_a.get_component<1>().set_mu(0.1);

  auto new_dist = make_mixture(Laplace<double>(-0.1, 1),
      Laplace<double>(0.1, 1));

  dist.MLE(samples);
  dist_a.MLE(samples);
  new_dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);
  double likelihood2_a = dist_a.log_likelihood(samples);
  double new_likelihood = new_dist.log_likelihood(samples);

  EXPECT_DOUBLE_EQ(likelihood2, likelihood2_a);
  EXPECT_GE(likelihood2, likelihood1);
  EXPECT_GE(likelihood2, new_likelihood);
}

TEST(MixtureTest, LikelihoodLaplace) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  Laplace<double> dist1(-10, 1);
  Laplace<double> dist2(10, 1);
  AsymmetricLaplace<double> dist3(0.5, -10, 1);
  AsymmetricLaplace<double> dist4(0.5, 10, 1);
  dist3.fix_p(true);
  dist4.fix_p(true);
  auto dist = make_mixture(dist1, dist2);
  auto dist_a = make_mixture(dist3, dist4);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);
  double likelihood1_a = dist_a.log_likelihood(samples);
  EXPECT_DOUBLE_EQ(likelihood1, likelihood1_a);

  dist.get_component<0>().set_mu(-0.1);
  dist.get_component<1>().set_mu(0.1);
  dist_a.get_component<0>().set_mu(-0.1);
  dist_a.get_component<1>().set_mu(0.1);

  auto new_dist = make_mixture(Laplace<double>(-0.1, 1),
      Laplace<double>(0.1, 1));

  dist.MLE(samples);
  dist_a.MLE(samples);
  new_dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);
  double likelihood2_a = dist_a.log_likelihood(samples);
  double new_likelihood = new_dist.log_likelihood(samples);

  EXPECT_DOUBLE_EQ(likelihood2, likelihood2_a);
  EXPECT_GE(likelihood2, likelihood1);
  EXPECT_GE(likelihood2, new_likelihood);
}

TEST(MixtureTest, LikelihoodAsymmetricNormal) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricNormal<double> dist1(0.5, -10, 1);
  AsymmetricNormal<double> dist2(0.5, 10, 1);
  auto dist = make_mixture(dist1, dist2);
  dist1.fix_p(true);
  dist2.fix_p(true);
  auto dist_a = make_mixture(dist1, dist2);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);
  double likelihood1_a = dist_a.log_likelihood(samples);
  EXPECT_DOUBLE_EQ(likelihood1, likelihood1_a);

  dist.get_component<0>().set_mu(-0.1);
  dist.get_component<1>().set_mu(0.1);
  dist_a.get_component<0>().set_mu(-0.1);
  dist_a.get_component<1>().set_mu(0.1);

  auto new_dist = make_mixture(Laplace<double>(-0.1, 1),
      Laplace<double>(0.1, 1));

  dist.get_component<0>().fix_p(true);
  dist.get_component<1>().fix_p(true);
  dist.MLE(samples);
  dist.get_component<0>().fix_p(false);
  dist.get_component<1>().fix_p(false);
  dist.MLE(samples);

  dist_a.MLE(samples);
  new_dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);
  double likelihood2_a = dist_a.log_likelihood(samples);
  double new_likelihood = new_dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood2_a);
  EXPECT_GE(likelihood2, likelihood1);
  EXPECT_GE(likelihood2, new_likelihood);
}

TEST(MixtureTest, LikelihoodAsymmetricLaplace) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 100;
  AsymmetricLaplace<double> dist1(0.5, -10, 1);
  AsymmetricLaplace<double> dist2(0.5, 10, 1);
  auto dist = make_mixture(dist1, dist2);
  dist1.fix_p(true);
  dist2.fix_p(true);
  auto dist_a = make_mixture(dist1, dist2);
  Array<double> samples;
  dist.sample(samples, n_samples, rng);

  double likelihood1 = dist.log_likelihood(samples);
  double likelihood1_a = dist_a.log_likelihood(samples);
  EXPECT_DOUBLE_EQ(likelihood1, likelihood1_a);

  dist.get_component<0>().set_mu(-0.1);
  dist.get_component<1>().set_mu(0.1);
  dist_a.get_component<0>().set_mu(-0.1);
  dist_a.get_component<1>().set_mu(0.1);

  auto new_dist = make_mixture(Normal<double>(-0.1, 1),
      Normal<double>(0.1, 1));

  dist.get_component<0>().fix_p(true);
  dist.get_component<1>().fix_p(true);
  dist.MLE(samples);
  dist.get_component<0>().fix_p(false);
  dist.get_component<1>().fix_p(false);
  dist.MLE(samples);

  dist_a.MLE(samples);
  new_dist.MLE(samples);

  double likelihood2 = dist.log_likelihood(samples);
  double likelihood2_a = dist_a.log_likelihood(samples);
  double new_likelihood = new_dist.log_likelihood(samples);

  EXPECT_GE(likelihood2, likelihood2_a);
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
