#ifndef __PROBABILITY_DISTRIBUTIONS__LAPLACE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__LAPLACE_IMPL_HPP__

#include "laplace.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <algorithm>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class T>
  Laplace<T>::Laplace(T mu, T b):
    fixed_mu_(false),
    fixed_b_(false),
    consider_sorted_(false) {
      set_mu(mu);
      set_b(b);
    }

  template <class T>
  template <class RNG>
  void Laplace<T>::sample(MA::Array<T>& samples, size_t n_samples, RNG& rng)
  const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = 1;
    samples.resize(size);

    boost::random::uniform_smallint<int> dist1(0, 1);
    boost::random::exponential_distribution<T> dist2(b_);

    MA::Slice<T> slice(samples, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::Array<T> s = slice.get_element(j);
      if (dist1(rng))
        s(0) = mu_ + dist2(rng);
      else
        s(0) = mu_ - dist2(rng);
    }
  }

  template <class T>
  T Laplace<T>::log_likelihood(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) const {
    check_data_and_weight(data, weight);

    T ll = 0;
    T b_likelihood = std::log(2*b_);

    MA::ConstSlice<T> slice(data, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::ConstArray<T> const& sample = slice.get_element(j);
      T w = weight(j);
      T s = sample(0);
      T local_likelihood = std::abs(s - mu_) * inv_b_;
      local_likelihood += b_likelihood;
      ll -= w * local_likelihood;
    }

    return ll;
  }

  template <class T>
  void Laplace<T>::MLE(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) {
    check_data_and_weight(data, weight);

    if (!fixed_mu_) {
      std::vector<size_t> sorted_indexes(data.size()[0]);
      std::generate_n(sorted_indexes.begin(), data.size()[0],
          []() { static size_t counter = 0; return counter++; });

      if (!consider_sorted_) {
        std::sort(sorted_indexes.begin(), sorted_indexes.end(),
            [&](size_t i, size_t j) { return data(i,0) <  data(j,0); });
      }

      T total_sum = 0;
      std::vector<T> p_n(data.size()[0]);
      for (size_t j = 0; j < data.size()[0]; j++) {
        T w = weight(sorted_indexes[j]);
        total_sum += w;
        p_n[j] = total_sum - w/2;
      }

      for (size_t j = 0; j < data.size()[0]; j++)
        p_n[j] /= total_sum;

      if (p_n[0] >= 0.5)
        set_mu(data(0,0));
      else if (p_n[p_n.size()-1] <= 0.5)
        set_mu(data(p_n.size()-1,0));
      else {
        for (size_t j = 1; j < data.size()[0]-1; j++)
          if (p_n[j] <= 0.5) {
            T scale = (0.5 - p_n[j])/(p_n[j+1] - p_n[j]);
            T offset = data(sorted_indexes[j], 0);
            set_mu(offset + scale * (data(sorted_indexes[j+1],0) -
                  data(sorted_indexes[j],0)));
          }
      }
    }

    if (!fixed_b_) {
      T sum_0 = 0, sum_1 = 0;
      MA::ConstSlice<T> slice(data, 0);
      for (size_t j = 0; j < slice.total_left_size(); j++) {
        MA::ConstArray<T> const& sample = slice.get_element(j);
        T w = weight(j);
        sum_0 += w;
        sum_1 += w*std::abs(sample(0) - mu_);
      }

      set_b(sum_1 / sum_0);
    }
  }

  template <class T>
  void Laplace<T>::check_data_and_weight(MA::ConstArray<T> const& data,
      MA::ConstArray<T> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
