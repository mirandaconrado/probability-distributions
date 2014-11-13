#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_DISTRIBUTION_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_DISTRIBUTION_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class Dist, class D, class W = D, class T = W>
  class AsymmetricDistribution: public Distribution<D,W,T> {
    public:
      AsymmetricDistribution(T p, T mu, T eps = 1e-4, T tol = 1e-6);

      static constexpr unsigned int sample_size = 1;

      void fix_p(bool fixed = true) { fixed_p_ = fixed; }
      void fix_mu(bool fixed = true) { fixed_mu_ = fixed; }
      void set_p(T p);
      void set_mu(T mu) { mu_ = mu; }
      T get_p() const { return p_; }
      T get_mu() const { return mu_; }

      template <class RNG>
      void sample(MA::Array<D>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<D,W,T>::log_likelihood;
      T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

      bool require_sorted() const { return true; }

    protected:
      void init();

      T fix_step(T p, T step) const;

      T compute_p_derivative(T p, T ll, MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);

      virtual void init_MLE(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes) { }
      virtual void end_MLE() { }

      virtual void updated_p() { }
      virtual T constant_likelihood() const { return 0; }
      virtual T negative_ll(T s, T mu) const = 0;
      virtual T positive_ll(T s, T mu) const = 0;
      virtual void set_parameter_vector(std::vector<T> const& p) = 0;
      virtual std::vector<T> get_parameter_vector() const = 0;

      virtual void MLE_fixed_p(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes) = 0;

      virtual void check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      bool fixed_mu_, fixed_p_;
      T mu_, p_, eps_, tol_;
  };
};

#include "asymmetric_distribution_impl.hpp"

#endif
