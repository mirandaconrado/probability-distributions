#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_HPP__

#include "asymmetric_distribution.hpp"

#include <boost/random/exponential_distribution.hpp>

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class AsymmetricLaplace:
    public AsymmetricDistribution<AsymmetricLaplace<D,W,T>,D,W,T> {
    public:
      AsymmetricLaplace(T p, T mu, T lambda);

      static constexpr unsigned int sample_size = 1;

      void fix_lambda(bool fixed = true) { fixed_lambda_ = fixed; }
      void set_lambda(T lambda) { assert(lambda > 0); lambda_ = lambda; }
      T get_lambda() const { return lambda_; }

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
          MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes);
      void end_MLE();
      void updated_p();

      void MLE_fixed_p(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);

      bool fixed_lambda_;
      T lambda_, alpha_, alpha_inv_;
      std::vector<T> percentile_vector_;
  };
};

#include "asymmetric_laplace_impl.hpp"

#endif
