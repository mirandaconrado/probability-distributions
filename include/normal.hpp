#ifndef __PROBABILITY_DISTRIBUTIONS__NORMAL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__NORMAL_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class T>
  class Normal: public Distribution<T> {
    public:
      Normal(T mean, T sigma);

      void fix_mean(bool fixed = true) { fixed_mean_ = fixed; }
      void fix_sigma(bool fixed = true) { fixed_sigma_ = fixed; }
      void set_mean(T mean) { mean_ = mean; }
      void set_sigma(T sigma) { sigma_ = sigma, inv_sigma2_ = 1/(2*sigma*sigma); }
      T get_mean() const { return mean_; }
      T get_sigma() const { return sigma_; }

      template <class RNG>
      void sample(MA::Array<T>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<T>::log_likelihood;
      T log_likelihood(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) const;

      using Distribution<T>::MLE;
      void MLE(MA::ConstArray<T> const& data, MA::ConstArray<T> const& weight);

    private:
      void check_data_and_weight(MA::ConstArray<T> const& data,
          MA::ConstArray<T> const& weight) const;

      bool fixed_mean_, fixed_sigma_;
      T mean_, sigma_, inv_sigma2_;
  };
};

#include "normal_impl.hpp"

#endif
