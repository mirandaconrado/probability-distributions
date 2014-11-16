#ifndef __PROBABILITY_DISTRIBUTIONS__NORMAL_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__NORMAL_IMPL_HPP__

#include "normal.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/normal_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  Normal<D,W,T>::Normal(T mu, T sigma):
    fixed_mu_(false),
    fixed_sigma_(false) {
      set_mu(mu);
      set_sigma(sigma);
    }

  template <class D, class W, class T>
  template <class RNG>
  void Normal<D,W,T>::sample(MA::Array<D>& samples, size_t n_samples, RNG& rng)
  const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::normal_distribution<T> dist(mu_, sigma_);

    D* ptr = samples.get_pointer();

    for (size_t j = 0; j < n_samples; j++)
      ptr[j] = dist(rng);
  }

  template <class D, class W, class T>
  T Normal<D,W,T>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    D const* ptr = data.get_pointer();

    T ll = 0;
    T sigma_likelihood = std::log(2*M_PI*sigma_*sigma_)/2;

    for (size_t j = 0; j < data.total_size(); j++) {
      T w = weight(j);
      T s = ptr[j];
      T local_likelihood = s - mu_;
      local_likelihood *= local_likelihood;
      local_likelihood *= inv_sigma2_;
      local_likelihood += sigma_likelihood;
      ll -= w * local_likelihood;
    }

    return ll;
  }

  template <class D, class W, class T>
  void Normal<D,W,T>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);

    D const* ptr = data.get_pointer();

    T sum_0 = 0, sum_1 = 0, sum_2 = 0;
    for (size_t j = 0; j < data.total_size(); j++) {
      T w = weight(j);
      T s = ptr[j];
      sum_0 += w;
      sum_1 += w*s;
      sum_2 += w*s*s;
    }

    if (!fixed_mu_)
      set_mu(sum_1/sum_0);
    if (!fixed_sigma_)
      set_sigma(std::sqrt((sum_2 - 2*mu_*sum_1 + mu_*mu_*sum_0)/sum_0));
  }

  template <class D, class W, class T>
  void Normal<D,W,T>::check_data_and_weight(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
