#ifndef __PROBABILITY_DISTRIBUTIONS__LAPLACE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__LAPLACE_IMPL_HPP__

#include "laplace.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/uniform_smallint.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  Laplace<D,W,T>::Laplace(T mu, T b):
    fixed_mu_(false),
    fixed_b_(false) {
      set_mu(mu);
      set_b(b);
    }

  template <class D, class W, class T>
  template <class RNG>
  void Laplace<D,W,T>::sample(MA::Array<D>& samples, size_t n_samples, RNG& rng)
  const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::uniform_smallint<int> dist1(0, 1);
    boost::random::exponential_distribution<T> dist2(b_);

    D* ptr = samples.get_pointer();

    for (size_t j = 0; j < n_samples; j++) {
      if (dist1(rng))
        ptr[j] = mu_ + dist2(rng);
      else
        ptr[j] = mu_ - dist2(rng);
    }
  }

  template <class D, class W, class T>
  T Laplace<D,W,T>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    D const* ptr = data.get_pointer();

    T ll = 0;
    T b_likelihood = std::log(2*b_);

    for (size_t j = 0; j < data.total_size(); j++) {
      T w = weight(j);
      T s = ptr[j];
      T local_likelihood = std::abs(s - mu_) * inv_b_;
      local_likelihood += b_likelihood;
      ll -= w * local_likelihood;
    }

    return ll;
  }

  template <class D, class W, class T>
  void Laplace<D,W,T>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);
    assert(data.size()[0] == indexes.size());

    D const* ptr = data.get_pointer();

    if (!fixed_mu_)
      set_mu(Distribution<D,W,T>::get_percentile(0.5, data, weight, indexes));

    if (!fixed_b_) {
      T sum_0 = 0, sum_1 = 0;
      for (size_t j = 0; j < data.total_size(); j++) {
        T w = weight(j);
        sum_0 += w;
        sum_1 += w*std::abs(ptr[j] - mu_);
      }

      set_b(sum_1 / sum_0);
    }
  }

  template <class D, class W, class T>
  void Laplace<D,W,T>::check_data_and_weight(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
