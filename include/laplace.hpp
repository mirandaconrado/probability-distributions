#ifndef __PROBABILITY_DISTRIBUTIONS__LAPLACE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__LAPLACE_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class Laplace: public Distribution<D,W,T> {
    public:
      Laplace(T mu, T b);

      void fix_mu(bool fixed = true) { fixed_mu_ = fixed; }
      void fix_b(bool fixed = true) { fixed_b_ = fixed; }
      void set_mu(T mu) { mu_ = mu; }
      void set_b(T b) { assert(b > 0); b_ = b, inv_b_ = 1/b; }
      T get_mu() const { return mu_; }
      T get_b() const { return b_; }

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

      bool fixed_mu_, fixed_b_;
      T mu_, b_, inv_b_;
  };
};

#include "laplace_impl.hpp"

#endif
