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
    auto data_size = data.size()[0];

    if (!fixed_mu_) {
      T total_sum = 0;
      std::vector<T> p_n(data_size);
      for (size_t j = 0; j < data_size; j++) {
        T w = weight(indexes[j]);
        total_sum += w;
        p_n[j] = total_sum - w/2;
      }

      for (size_t j = 0; j < data_size; j++)
        p_n[j] /= total_sum;

      if (p_n[0] >= 0.5)
        set_mu(ptr[indexes[0]]);
      else if (p_n[p_n.size()-1] <= 0.5)
        set_mu(ptr[indexes[p_n.size()-1]]);
      else {
        for (size_t j = 1; j < data_size-1; j++)
          if (p_n[j] <= 0.5) {
            T scale = (0.5 - p_n[j])/(p_n[j+1] - p_n[j]);
            T offset = ptr[indexes[j]];
            set_mu(offset + scale * (ptr[indexes[j+1]] - ptr[indexes[j]]));
          }
      }
    }

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
