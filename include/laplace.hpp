#ifndef __PROBABILITY_DISTRIBUTIONS__LAPLACE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__LAPLACE_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class Laplace: public Distribution<D,W,T> {
    public:
      Laplace(T mu, T lambda);

      static constexpr unsigned int sample_size = 1;

      void fix_mu(bool fixed = true) { fixed_mu_ = fixed; }
      void fix_lambda(bool fixed = true) { fixed_lambda_ = fixed; }
      void set_mu(T mu) { mu_ = mu; }
      void set_lambda(T lambda) { assert(lambda > 0); lambda_ = lambda; }
      T get_mu() const { return mu_; }
      T get_lambda() const { return lambda_; }

      template <class RNG>
      void sample(MA::Array<D>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<D,W,T>::log_likelihood;
      T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

      bool require_sorted() const { return true; }

    private:
      void check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      bool fixed_mu_, fixed_lambda_;
      T mu_, lambda_;
  };
};

#include "laplace_impl.hpp"

#endif
