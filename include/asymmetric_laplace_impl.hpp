#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_IMPL_HPP__

#include "asymmetric_laplace.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  AsymmetricLaplace<D,W,T>::AsymmetricLaplace(T p, T mu, T lambda):
    base_class(p, mu),
    fixed_lambda_(false) {
      base_class::init();
      set_lambda(lambda);
    }

  template <class D, class W, class T>
  boost::random::exponential_distribution<T>
  AsymmetricLaplace<D,W,T>::create_gamma_plus() const {
    return boost::random::exponential_distribution<T>(lambda_ * alpha_);
  }

  template <class D, class W, class T>
  boost::random::exponential_distribution<T>
  AsymmetricLaplace<D,W,T>::create_gamma_minus() const {
    return boost::random::exponential_distribution<T>(lambda_ * alpha_inv_);
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::constant_likelihood() const {
    return std::log(lambda_);
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::negative_ll(T s, T mu) const {
    T neg_const = lambda_ * alpha_inv_;
    return (s - mu) * neg_const;
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::positive_ll(T s, T mu) const {
    T pos_const = -lambda_ * alpha_;
    return (s - mu) * pos_const;
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::init_MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes) {
    percentile_vector_ = Distribution<D,W,T>::build_percentile_vector(weight,
        indexes);
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::end_MLE() {
    percentile_vector_.clear();
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::updated_p() {
    alpha_ = std::sqrt(base_class::p_/(1-base_class::p_));
    alpha_inv_ = 1/alpha_;
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::MLE_fixed_p(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    D const* ptr = data.get_pointer();

    if (!base_class::fixed_mu_)
      base_class::set_mu(Distribution<D,W,T>::get_percentile(base_class::p_,
            data, weight, indexes, percentile_vector_));

    if (!fixed_lambda_) {
      T sum_0 = 0, sum_1_pos = 0, sum_1_neg = 0;
      for (size_t j = 0; j < data.total_size(); j++) {
        T w = weight(j);
        sum_0 += w;
        if (ptr[j] < base_class::mu_)
          sum_1_neg += ptr[j] - base_class::mu_;
        else
          sum_1_pos += ptr[j] - base_class::mu_;
      }

      set_lambda(sum_0 / (alpha_ * sum_1_pos - alpha_inv_ * sum_1_neg));
    }
  }
};

#endif
