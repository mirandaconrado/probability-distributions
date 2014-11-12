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
    fixed_p_(false),
    fixed_mu_(false),
    fixed_lambda_(false) {
      set_p(p);
      set_mu(mu);
      set_lambda(lambda);
    }

  template <class D, class W, class T>
  template <class RNG>
  void AsymmetricLaplace<D,W,T>::sample(MA::Array<D>& samples, size_t n_samples,
      RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::uniform_real_distribution<T> dist(0, 1);
    boost::random::exponential_distribution<T> gamma_plus(lambda_ * alpha_),
      gamma_minus(lambda_ * alpha_inv_);

    D* ptr = samples.get_pointer();

    for (size_t j = 0; j < n_samples; j++) {
      if (dist(rng) < p_)
        ptr[j] = mu_ - gamma_minus(rng);
      else
        ptr[j] = mu_ + gamma_plus(rng);
    }
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    D const* ptr = data.get_pointer();

    T ll = 0;
    T lambda_likelihood = std::log(lambda_) + (std::log(p_) + std::log(1-p_))/2;

    T neg_const = lambda_ * alpha_inv_, pos_const = -lambda_ * alpha_;

    for (size_t j = 0; j < data.total_size(); j++) {
      T w = weight(j);
      T s = ptr[j];
      T local_likelihood = lambda_likelihood;
      if (s < mu_)
        local_likelihood += (s - mu_) * neg_const;
      else
        local_likelihood += (s - mu_) * pos_const;
      ll += w * local_likelihood;
    }

    return ll;
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);
    assert(data.size()[0] == indexes.size());

    percentile_vector_ = Distribution<D,W,T>::build_percentile_vector(weight,
        indexes);

    if (fixed_p_)
      MLE_fixed_p(data, weight, indexes);
    else {
      T eps = 1e-4;
      T ll = log_likelihood(data, weight);
      T diff = compute_p_derivative(p_, ll, eps, data, weight, indexes);

      if (isnan(diff))
        return;

      T step = fix_step(p_, (diff > 0) ? 1e-2 : -1e-2);

      T best_p = p_, best_mu = mu_, best_lambda = lambda_;
      T best_ll = ll;

      T tol = 1e-6;
      while (std::abs(step) > tol) {
        set_p(best_p + step);
        MLE_fixed_p(data, weight, indexes);
        T new_ll = log_likelihood(data, weight);

        if (new_ll >= best_ll) {
          best_p = p_;
          best_mu = mu_;
          best_lambda = lambda_;
          best_ll = new_ll;

          diff = compute_p_derivative(best_p, best_ll, eps, data, weight,
              indexes);
          if (isnan(diff))
            return;
          step = fix_step(p_,
              ((diff > 0) ? std::abs(step) : -std::abs(step)) * 1.1);
        }
        else
          step *= 0.5;
      }

      set_p(best_p);
      set_mu(best_mu);
      set_lambda(best_lambda);
    }

    percentile_vector_.clear();
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::fix_step(T p, T step) const {
    while (p_ + step <= 0 || p_ + step >= 1)
      step *= 0.99;
    return step;
  }

  template <class D, class W, class T>
  T AsymmetricLaplace<D,W,T>::compute_p_derivative(T p, T ll, T eps,
      MA::ConstArray<D> const& data, MA::ConstArray<D> const& weight,
      std::vector<size_t> const& indexes) {
    T ll_pos, ll_neg;
    if (p_ + eps <= 1 && p_ - eps >= 0) {
      set_p(p_ + eps);
      MLE_fixed_p(data, weight, indexes);
      ll_pos = log_likelihood(data, weight);
      set_p(p_ - eps);
      MLE_fixed_p(data, weight, indexes);
      ll_neg = log_likelihood(data, weight);
      return (ll_pos - ll_neg) / (2*eps);
    }
    else if (p_ + eps <= 1) {
      set_p(p_ + eps);
      MLE_fixed_p(data, weight, indexes);
      ll_pos = log_likelihood(data, weight);
      return (ll_pos - ll) / eps;
    }
    else if (p_ - eps >= 0) {
      set_p(p_ - eps);
      MLE_fixed_p(data, weight, indexes);
      ll_neg = log_likelihood(data, weight);
      return (ll - ll_neg) / eps;
    }
    else
      return NAN;
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::MLE_fixed_p(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    D const* ptr = data.get_pointer();

    if (!fixed_mu_)
      set_mu(Distribution<D,W,T>::get_percentile(p_, data, weight, indexes,
            percentile_vector_));

    if (!fixed_lambda_) {
      T sum_0 = 0, sum_1_pos = 0, sum_1_neg = 0;
      for (size_t j = 0; j < data.total_size(); j++) {
        T w = weight(j);
        sum_0 += w;
        if (ptr[j] < mu_)
          sum_1_neg += ptr[j] - mu_;
        else
          sum_1_pos += ptr[j] - mu_;
      }

      set_lambda(sum_0 / (alpha_ * sum_1_pos - alpha_inv_ * sum_1_neg));
    }
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::check_data_and_weight(
      MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }

  template <class D, class W, class T>
  void AsymmetricLaplace<D,W,T>::compute_alpha() {
    alpha_ = std::sqrt(p_/(1-p_));
    alpha_inv_ = 1/alpha_;
  }
};

#endif
