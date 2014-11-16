#ifndef __PROBABILITY_DISTRIBUTIONS__NORMAL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__NORMAL_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class Normal: public Distribution<D,W,T> {
    public:
      Normal(T mu, T sigma);

      static constexpr unsigned int sample_size = 1;

      void fix_mu(bool fixed = true) { fixed_mu_ = fixed; }
      void fix_sigma(bool fixed = true) { fixed_sigma_ = fixed; }
      void set_mu(T mu) { mu_ = mu; }
      void set_sigma(T sigma) { assert(sigma > 0); sigma_ = sigma;
        inv_sigma2_ = 1/(2*sigma*sigma); }
      T get_mu() const { return mu_; }
      T get_sigma() const { return sigma_; }
      Normal<D,W,T> const&
        operator=(Normal<D,W,T> const& other) {
          set_mu(other.get_mu());
          set_sigma(other.get_sigma());
          fixed_mu_ = other.fixed_mu_;
          fixed_sigma_ = other.fixed_sigma_;
          return *this;
        }

      template <class RNG>
      void sample(MA::Array<D>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<D,W,T>::log_likelihood;
      T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

    private:
      void check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      bool fixed_mu_, fixed_sigma_;
      T mu_, sigma_, inv_sigma2_;
  };
};

#include "normal_impl.hpp"

#endif
