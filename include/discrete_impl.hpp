#ifndef __PROBABILITY_DISTRIBUTIONS__DISCRETE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISCRETE_IMPL_HPP__

#include "discrete.hpp"

#include "const_slice.hpp"
#include "slice.hpp"

#include <boost/random/discrete_distribution.hpp>
#include <cmath>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  Discrete<D,W,T>::Discrete(unsigned int K):
    p_(K, 1./K) {
      assert(K > 0);
    }

  template <class D, class W, class T>
  Discrete<D,W,T>::Discrete(std::vector<T> const& p):
    p_(p) {
      assert(p.size() > 0);
    }

  template <class D, class W, class T>
  template <class RNG>
  void Discrete<D,W,T>::sample(MA::Array<D>& samples, size_t n_samples,
      RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = p_.size();
    samples.resize(size);

    boost::random::discrete_distribution<unsigned int, T> dist(p_.begin(),
        p_.end());

    MA::Slice<T> slice(samples, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::Array<T> s = slice.get_element(j);

      unsigned int val = dist(rng);
      for (unsigned int i = 0; i < p_.size(); i++) {
        if (i == val)
          s(i) = 1;
        else
          s(i) = 0;
      }
    }
  }

  template <class D, class W, class T>
  T Discrete<D,W,T>::log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    T ll = 0;
    std::vector<T> log_p(p_);
    for (unsigned int i = 0; i < p_.size(); i++)
      log_p[i] = log(log_p[i]);

    MA::ConstSlice<T> slice(data, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::ConstArray<T> const& sample = slice.get_element(j);
      T w = weight(j);
      for (unsigned int i = 0; i < p_.size(); i++) {
        T s = sample(i);
        if (s != 0)
          ll += w * s * log_p[i];
      }
    }

    return ll;
  }

  template <class D, class W, class T>
  void Discrete<D,W,T>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);

    for (unsigned int i = 0; i < p_.size(); i++)
      p_[i] = 0;

    MA::ConstSlice<T> slice(data, 0);
    for (size_t j = 0; j < slice.total_left_size(); j++) {
      MA::ConstArray<T> const& sample = slice.get_element(j);
      T w = weight(j);
      for (unsigned int i = 0; i < p_.size(); i++)
        p_[i] += w * sample(i);
    }

    normalize();
  }

  template <class D, class W, class T>
  void Discrete<D,W,T>::sample_to_index(MA::Array<unsigned int>& indexes,
          MA::ConstArray<D> const& samples) const {
    assert(samples.size().size() == 2);
    assert(samples.size()[0] > 0);
    assert(samples.size()[1] == p_.size());

    MA::Size::SizeType size = samples.size();
    size[1] = 1;
    indexes.resize(size);

    MA::ConstSlice<T> samples_slice(samples, 0);
    MA::Slice<unsigned int> indexes_slice(indexes, 0);
    for (size_t j = 0; j < samples_slice.total_left_size(); j++) {
      MA::ConstArray<T> const& s = samples_slice.get_element(j);
      MA::Array<unsigned int> idx = indexes_slice.get_element(j);
      size_t i;
      for (i = 0; i < p_.size(); i++)
        if (s(i) == 1) {
          idx(0) = i;
          break;
        }
      assert(i < p_.size());
    }
  }

  template <class D, class W, class T>
  void Discrete<D,W,T>::index_to_sample(MA::Array<D>& samples,
          MA::ConstArray<unsigned int> const& indexes) const {
    assert(indexes.size().size() == 2);
    assert(indexes.size()[0] > 0);
    assert(indexes.size()[1] == 1);

    MA::Size::SizeType size = indexes.size();
    size[1] = p_.size();
    samples.resize(size);

    MA::Slice<T> samples_slice(samples, 0);
    MA::ConstSlice<unsigned int> indexes_slice(indexes, 0);
    for (size_t j = 0; j < indexes_slice.total_left_size(); j++) {
      MA::Array<T>& s = samples_slice.get_element(j);
      MA::ConstArray<unsigned int> const& idx = indexes_slice.get_element(j);
      unsigned int index = idx(0);
      assert(index < p_.size());
      for (size_t i = 0; i < p_.size(); i++) {
        if (i == index)
          s(i) = 1;
        else
          s(i) = 0;
      }
    }
  }

  template <class D, class W, class T>
  void Discrete<D,W,T>::check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == p_.size());
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }

  template <class D, class W, class T>
  void Discrete<D,W,T>::normalize() {
    T sum = 0;
    for (unsigned int i = 0; i < p_.size(); i++)
      sum += p_[i];

    for (unsigned int i = 0; i < p_.size(); i++)
      p_[i] /= sum;
  }
};

#endif
