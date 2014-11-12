#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_DISTRIBUTION_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_DISTRIBUTION_IMPL_HPP__

#include "asymmetric_distribution.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class Dist, class D, class W, class T>
  AsymmetricDistribution<Dist,D,W,T>::AsymmetricDistribution(T p, T mu, T eps,
      T tol):
    fixed_mu_(false),
    fixed_p_(false),
    mu_(mu),
    eps_(eps),
    tol_(tol) {
      set_mu(mu);
      set_p(p);
    }

  template <class Dist, class D, class W, class T>
  void AsymmetricDistribution<Dist,D,W,T>::init() {
    set_p(p_);
  }

  template <class Dist, class D, class W, class T>
  void AsymmetricDistribution<Dist,D,W,T>::set_p(T p) {
    assert(p > 0);
    assert(p < 1);
    p_ = p;
    static_cast<Dist*>(this)->updated_p();
  }

  template <class Dist, class D, class W, class T>
  template <class RNG>
  void AsymmetricDistribution<Dist,D,W,T>::sample(MA::Array<D>& samples,
      size_t n_samples, RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::uniform_real_distribution<T> dist(0, 1);
    auto gamma_plus = static_cast<Dist const*>(this)->create_gamma_plus();
    auto gamma_minus = static_cast<Dist const*>(this)->create_gamma_minus();

    D* ptr = samples.get_pointer();

    for (size_t j = 0; j < n_samples; j++) {
      if (dist(rng) < p_)
        ptr[j] = mu_ - gamma_minus(rng);
      else
        ptr[j] = mu_ + gamma_plus(rng);
    }
  }

  template <class Dist, class D, class W, class T>
  T AsymmetricDistribution<Dist,D,W,T>::log_likelihood(
      MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    D const* ptr = data.get_pointer();

    T ll = 0;
    T const_likelihood = static_cast<Dist const*>(this)->constant_likelihood() +
      (std::log(p_) + std::log(1-p_))/2;

    for (size_t j = 0; j < data.total_size(); j++) {
      T w = weight(j);
      T s = ptr[j];
      T local_likelihood = const_likelihood;
      if (s < mu_)
        local_likelihood += static_cast<Dist const*>(this)->negative_ll(s, mu_);
      else
        local_likelihood += static_cast<Dist const*>(this)->positive_ll(s, mu_);
      ll += w * local_likelihood;
    }

    return ll;
  }

  template <class Dist, class D, class W, class T>
  void AsymmetricDistribution<Dist,D,W,T>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);
    assert(data.size()[0] == indexes.size());

    static_cast<Dist*>(this)->init_MLE(data, weight, indexes);

    if (fixed_p_)
      static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
    else {
      T ll = log_likelihood(data, weight);
      T diff = compute_p_derivative(p_, ll, data, weight, indexes);

      if (isnan(diff))
        return;

      T step = fix_step(p_, (diff > 0) ? 1e-2 : -1e-2);

      T best_p = p_, best_mu = mu_;
      std::vector<T> best_parameters =
        static_cast<Dist*>(this)->get_parameter_vector();
      T best_ll = ll;

      while (std::abs(step) > tol_) {
        set_p(best_p + step);
        static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
        T new_ll = log_likelihood(data, weight);

        if (new_ll >= best_ll) {
          best_p = p_;
          best_mu = mu_;
          best_parameters = static_cast<Dist*>(this)->get_parameter_vector();
          best_ll = new_ll;

          diff = compute_p_derivative(best_p, best_ll, data, weight, indexes);
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
      static_cast<Dist*>(this)->set_parameter_vector(best_parameters);
    }

    static_cast<Dist*>(this)->end_MLE();
  }

  template <class Dist, class D, class W, class T>
  T AsymmetricDistribution<Dist,D,W,T>::fix_step(T p, T step) const {
    while (p_ + step <= 0 || p_ + step >= 1)
      step *= 0.99;
    return step;
  }

  template <class Dist, class D, class W, class T>
  T AsymmetricDistribution<Dist,D,W,T>::compute_p_derivative(T p, T ll,
      MA::ConstArray<D> const& data, MA::ConstArray<D> const& weight,
      std::vector<size_t> const& indexes) {
    T ll_pos, ll_neg;
    if (p_ + eps_ <= 1 && p_ - eps_ >= 0) {
      set_p(p_ + eps_);
      static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
      ll_pos = log_likelihood(data, weight);
      set_p(p_ - eps_);
      static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
      ll_neg = log_likelihood(data, weight);
      return (ll_pos - ll_neg) / (2*eps_);
    }
    else if (p_ + eps_ <= 1) {
      set_p(p_ + eps_);
      static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
      ll_pos = log_likelihood(data, weight);
      return (ll_pos - ll) / eps_;
    }
    else if (p_ - eps_ >= 0) {
      set_p(p_ - eps_);
      static_cast<Dist*>(this)->MLE_fixed_p(data, weight, indexes);
      ll_neg = log_likelihood(data, weight);
      return (ll - ll_neg) / eps_;
    }
    else
      return NAN;
  }

  template <class Dist, class D, class W, class T>
  void AsymmetricDistribution<Dist,D,W,T>::check_data_and_weight(
      MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
