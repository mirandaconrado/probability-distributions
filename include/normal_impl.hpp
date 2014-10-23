#ifndef __PROBABILITY_DISTRIBUTIONS__NORMAL_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__NORMAL_IMPL_HPP__

#include "normal.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/normal_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class T>
  Normal<T>::Normal(T mean, T sigma):
    mean_(mean),
    sigma_(sigma),
    inv_sigma2_(1/(2*sigma*sigma)) {
      assert(sigma > 0);
    }

  template <class T>
  template <class RNG>
  void Normal<T>::sample(MA::Array<T>& samples, size_t n_samples, RNG& rng)
  const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::normal_distribution<T> dist(mean_, sigma_);

    MA::Slice<T> slice(samples, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::Array<T> s = slice.get_element(j);
      s(0) = dist(rng);
    }
  }

  template <class T>
  T Normal<T>::log_likelihood(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) const {
    check_data_and_weight(data, weight);

    T ll = 0;
    T sigma_likelihood = log(2*M_PI*sigma_*sigma_)/2;

    MA::ConstSlice<T> slice(data, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::ConstArray<T> const& sample = slice.get_element(j);
      T w = weight(j);
      T s = sample(0);
      T local_likelihood = s - mean_;
      local_likelihood *= local_likelihood;
      local_likelihood *= inv_sigma2_;
      local_likelihood += sigma_likelihood;
      ll -= w * local_likelihood;
    }

    return ll;
  }

  template <class T>
  void Normal<T>::MLE(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) {
    check_data_and_weight(data, weight);

    T sum_0 = 0, sum_1 = 0, sum_2 = 0;
    MA::ConstSlice<T> slice(data, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::ConstArray<T> const& sample = slice.get_element(j);
      T w = weight(j);
      sum_0 += w;
      sum_1 += w*sample(0);
      sum_2 += w*sample(0)*sample(0);
    }

    set_mean(sum_1/sum_0);
    set_sigma(sqrt((sum_2 - 2*mean_*sum_1 + mean_*mean_*sum_0)/sum_0));
  }

  template <class T>
  void Normal<T>::check_data_and_weight(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
