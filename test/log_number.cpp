#include "log_number.hpp"

#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

using namespace ProbabilityDistributions;

TEST(LogNumberTest, DefaultConstruction) {
  LogNumber val;
  EXPECT_EQ(0, val.to_double());
}

TEST(LogNumberTest, CopyConstruction) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double val = dist(rng);
    LogNumber val2(val);
    EXPECT_NEAR(val, val2.to_double(), 1e-10);
  }
}

TEST(LogNumberTest, Assignment) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double val = dist(rng);
    LogNumber val2;
    val2 = val;
    EXPECT_NEAR(val, val2.to_double(), 1e-10);
  }
}

TEST(LogNumberTest, FromLog) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double val = dist(rng);
    LogNumber val2;
    val2.from_log(val);
    EXPECT_NEAR(std::exp(val), val2.to_double(), 1e-10);
  }
}

TEST(LogNumberTest, Sum) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double v1 = dist(rng), v2 = dist(rng);
    LogNumber lv1(v1), lv2(v2);
    EXPECT_NEAR(v1+v2, (lv1+lv2).to_double(), 1e-10);
  }
}

TEST(LogNumberTest, Subtraction) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double v1 = dist(rng), v2 = dist(rng);
    LogNumber lv1(v1), lv2(v2);
    EXPECT_NEAR(v1-v2, (lv1-lv2).to_double(), 1e-10);
  }
}

TEST(LogNumberTest, Multiplication) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double v1 = dist(rng), v2 = dist(rng);
    LogNumber lv1(v1), lv2(v2);
    EXPECT_NEAR(v1*v2, (lv1*lv2).to_double(), 1e-10);
  }
}

TEST(LogNumberTest, Division) {
  boost::random::mt19937 rng;
  const unsigned int n_samples = 1000;
  boost::random::uniform_real_distribution<> dist(-5,5);

  for (unsigned int i = 0; i < n_samples; i++) {
    double v1 = dist(rng), v2 = dist(rng);
    LogNumber lv1(v1), lv2(v2);
    EXPECT_NEAR(v1/v2, (lv1/lv2).to_double(), 1e-10);
  }
}
