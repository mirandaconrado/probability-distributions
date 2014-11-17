#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_HPP__

#include "asymmetric_distribution.hpp"
#include "laplace.hpp"

#include <boost/random/exponential_distribution.hpp>

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class AsymmetricLaplace:
    public AsymmetricDistribution<AsymmetricLaplace<D,W,T>,D,W,T> {
    public:
      AsymmetricLaplace(T p, T mu, T lambda);

      static constexpr unsigned int sample_size = 1;

      void fix_lambda(bool fixed = true) { fixed_lambda_ = fixed; }
      bool is_lambda_fixed() const { return fixed_lambda_; }
      void set_lambda(T lambda) { assert(lambda > 0); lambda_ = lambda; }
      T get_lambda() const { return lambda_; }

      AsymmetricLaplace<D,W,T> const&
        operator=(AsymmetricLaplace<D,W,T> const& other) {
          base_class::set_p(other.get_p());
          base_class::set_mu(other.get_mu());
          set_lambda(other.get_lambda());
          base_class::fixed_p_ = other.is_p_fixed();
          base_class::fixed_mu_ = other.is_mu_fixed();
          fixed_lambda_ = other.is_lambda_fixed();
          return *this;
        }
      AsymmetricLaplace<D,W,T> const&
        operator=(Laplace<D,W,T> const& other) {
          base_class::set_p(0.5);
          base_class::set_mu(other.get_mu());
          set_lambda(other.get_lambda());
          base_class::fixed_p_ = true;
          base_class::fixed_mu_ = other.is_mu_fixed();
          fixed_lambda_ = other.is_lambda_fixed();
          return *this;
        }

    private:
      typedef AsymmetricDistribution<AsymmetricLaplace<D,W,T>,D,W,T> base_class;
      friend class AsymmetricDistribution<AsymmetricLaplace<D,W,T>,D,W,T>;

      boost::random::exponential_distribution<T> create_gamma_plus() const;
      boost::random::exponential_distribution<T> create_gamma_minus() const;

      T constant_likelihood() const;
      T negative_ll(T s, T mu) const;
      T positive_ll(T s, T mu) const;
      void set_parameter_vector(std::vector<T> const& p) {
        assert(p.size() == 1); lambda_ = p[0]; }
      std::vector<T> get_parameter_vector() const { return {lambda_}; }

      void init_MLE(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);
      void end_MLE();
      void updated_p();

      void MLE_fixed_p(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);

      bool fixed_lambda_;
      T lambda_, alpha_, alpha_inv_;
      std::vector<T> percentile_vector_;
      std::vector<T> pos_sum_all_0_, pos_sum_all_1_;
      std::vector<T> neg_sum_all_0_, neg_sum_all_1_;
  };
};

#include "asymmetric_laplace_impl.hpp"

#endif
