#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_NORMAL_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_NORMAL_IMPL_HPP__

#include "asymmetric_normal.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <cmath>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  AsymmetricNormal<D,W,T>::AsymmetricNormal(T p, T mu, T sigma):
    base_class(p, mu),
    fixed_sigma_(false) {
      base_class::init();
      set_sigma(sigma);
    }

  template <class D, class W, class T>
  auto AsymmetricNormal<D,W,T>::create_gamma_plus() const -> TruncatedNormal {
    return TruncatedNormal(sigma_/alpha_);
  }

  template <class D, class W, class T>
  auto AsymmetricNormal<D,W,T>::create_gamma_minus() const -> TruncatedNormal {
    return TruncatedNormal(sigma_*alpha_);
  }

  template <class D, class W, class T>
  T AsymmetricNormal<D,W,T>::constant_likelihood() const {
    return -std::log(sigma_) -std::log(M_PI/2)/2;
  }

  template <class D, class W, class T>
  T AsymmetricNormal<D,W,T>::negative_ll(T s, T mu) const {
    T neg_const = -1/(sigma2_*alpha2_);
    T diff = s - mu;
    return diff * diff * neg_const;
  }

  template <class D, class W, class T>
  T AsymmetricNormal<D,W,T>::positive_ll(T s, T mu) const {
    T pos_const = -alpha2_/(sigma2_);
    T diff = s - mu;
    return diff * diff * pos_const;
  }

  template <class D, class W, class T>
  void AsymmetricNormal<D,W,T>::init_MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    D const* ptr = data.get_pointer();

    neg_sum_all_0_.resize(data.total_size(), 0);
    neg_sum_all_1_.resize(data.total_size(), 0);
    neg_sum_all_2_.resize(data.total_size(), 0);
    pos_sum_all_0_.resize(data.total_size(), 0);
    pos_sum_all_1_.resize(data.total_size(), 0);
    pos_sum_all_2_.resize(data.total_size(), 0);

    T w, s;
    size_t n_data = data.total_size();

    for (size_t i = 1; i < n_data; i++) {
      w = weight(indexes[i-1]);
      s = ptr[indexes[i-1]];
      neg_sum_all_0_[i] = neg_sum_all_0_[i-1] + w;
      neg_sum_all_1_[i] = neg_sum_all_1_[i-1] + w * s;
      neg_sum_all_2_[i] = neg_sum_all_2_[i-1] + w * s * s;
    }

    for (size_t i = n_data-1; i > 0; i--) {
      w = weight(indexes[i]);
      s = ptr[indexes[i]];
      pos_sum_all_0_[i-1] = pos_sum_all_0_[i] + w;
      pos_sum_all_1_[i-1] = pos_sum_all_1_[i] + w * s;
      pos_sum_all_2_[i-1] = pos_sum_all_2_[i] + w * s * s;
    }
  }

  template <class D, class W, class T>
  void AsymmetricNormal<D,W,T>::end_MLE() {
    pos_sum_all_0_.clear();
    pos_sum_all_1_.clear();
    pos_sum_all_2_.clear();
    neg_sum_all_0_.clear();
    neg_sum_all_1_.clear();
    neg_sum_all_2_.clear();
  }

  template <class D, class W, class T>
  void AsymmetricNormal<D,W,T>::updated_p() {
    alpha_ = std::sqrt(base_class::p_/(1-base_class::p_));
    alpha_inv_ = 1/alpha_;
    alpha2_ = alpha_ * alpha_;
    alpha2_inv_ = alpha_inv_ * alpha_inv_;
  }

  template <class D, class W, class T>
  void AsymmetricNormal<D,W,T>::MLE_fixed_p(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    D const* ptr = data.get_pointer();

    if (!base_class::fixed_mu_) {
      if (data.total_size() == 1) {
        base_class::set_mu(ptr[0]);
      }
      else {
        for (size_t split = 0; split < data.total_size()-1; split++) {
          T mu = alpha2_inv_ * neg_sum_all_1_[split+1] +
            alpha2_ * pos_sum_all_1_[split];
          mu /= alpha2_inv_ * neg_sum_all_0_[split+1] +
            alpha2_ * pos_sum_all_0_[split];

          if (mu >= ptr[indexes[split]] && mu <= ptr[indexes[split+1]]) {
            base_class::set_mu(mu);
            break;
          }
        }
      }
    }

    if (!fixed_sigma_) {
      T& mu = base_class::mu_;

      size_t split;
      for (split = 0; split < data.total_size()-1; split++)
        if (mu >= ptr[indexes[split]] && mu <= ptr[indexes[split+1]])
          break;

      T sigma = 0;
      sigma += alpha2_inv_ * (neg_sum_all_2_[split+1] -
          2 * mu * neg_sum_all_1_[split+1] + mu * mu * neg_sum_all_0_[split+1]);
      sigma += alpha2_ * (pos_sum_all_2_[split] -
          2 * mu * pos_sum_all_1_[split] + mu * mu * pos_sum_all_0_[split]);
      sigma /= neg_sum_all_0_[split+1] + pos_sum_all_0_[split];
      set_sigma(std::sqrt(sigma));
    }
  }
};

#endif
