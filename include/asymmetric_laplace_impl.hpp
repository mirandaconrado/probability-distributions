#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_LAPLACE_IMPL_HPP__

#include "asymmetric_laplace.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

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
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    D const* ptr = data.get_pointer();

    neg_sum_all_0_.resize(data.total_size(), 0);
    neg_sum_all_1_.resize(data.total_size(), 0);
    pos_sum_all_0_.resize(data.total_size(), 0);
    pos_sum_all_1_.resize(data.total_size(), 0);

    T w, s;
    size_t n_data = data.total_size();

    for (size_t i = 1; i < n_data; i++) {
      w = weight(indexes[i-1]);
      s = ptr[indexes[i-1]];
      neg_sum_all_0_[i] = neg_sum_all_0_[i-1] + w;
      neg_sum_all_1_[i] = neg_sum_all_1_[i-1] + w * s;
    }

    for (size_t i = n_data-1; i > 0; i--) {
      w = weight(indexes[i]);
      s = ptr[indexes[i]];
      pos_sum_all_0_[i-1] = pos_sum_all_0_[i] + w;
      pos_sum_all_1_[i-1] = pos_sum_all_1_[i] + w * s;
    }
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::end_MLE() {
    pos_sum_all_0_.clear();
    pos_sum_all_1_.clear();
    neg_sum_all_0_.clear();
    neg_sum_all_1_.clear();
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

    if (!base_class::fixed_mu_) {
      if (data.total_size() == 1) {
        base_class::set_mu(ptr[0]);
      }
      else {
        if (pos_sum_all_0_[0]*alpha_ < neg_sum_all_0_[1]*alpha_inv_) {
          base_class::set_mu(ptr[indexes[0]]);
        }
        else if (pos_sum_all_0_[data.total_size()-2]*alpha_ >
            neg_sum_all_0_[data.total_size()-1]*alpha_inv_) {
          base_class::set_mu(ptr[indexes[data.total_size()-1]]);
        }
        else {
          size_t split;
          for (split = 1; split < data.total_size(); split++) {
            W sum_weight = pos_sum_all_0_[split-1]*alpha_ +
              neg_sum_all_0_[split]*alpha_inv_;

            W SN = sum_weight;
            W Sn = neg_sum_all_0_[split]*alpha_inv_;
            W p1 = (Sn - weight(indexes[split-1])*alpha_inv_/2)/SN;
            W p2 = (Sn + weight(indexes[split])*alpha_/2)/SN;
            if (p1 <= 0.5 && p2 >= 0.5) {
              D v1 = ptr[indexes[split-1]], v2 = ptr[indexes[split]];
              base_class::set_mu(v1 + (v2-v1)*(0.5-p1)/(p2-p1));
              break;
            }
          }
          assert(split < data.total_size());
        }
      }
    }

    if (!fixed_lambda_) {
      T& mu = base_class::mu_;

      size_t split;
      for (split = 0; split < data.total_size()-1; split++)
        if (mu >= ptr[indexes[split]] && mu <= ptr[indexes[split+1]])
          break;

      T lambda_inv = alpha_ * (pos_sum_all_1_[split] -
          mu * pos_sum_all_0_[split]);
      lambda_inv -= alpha_inv_ * (neg_sum_all_1_[split+1] -
          mu * neg_sum_all_0_[split+1]);

      set_lambda((neg_sum_all_0_[split+1] + pos_sum_all_0_[split])/lambda_inv);
    }
  }
};

#endif
