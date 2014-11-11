#ifndef __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_IMPL_HPP__

#include "distribution.hpp"

#include <algorithm>

namespace ProbabilityDistributions {
  template <class D, class W, class T>
  T Distribution<D,W,T>::log_likelihood(MA::ConstArray<D> const& data) const {
    assert(data.size().size() > 1);
    assert(data.size()[0] > 0);
    std::vector<W> weights(data.size()[0], 1);
    return log_likelihood(data, MA::ConstArray<W>({data.size()[0]},
          &weights[0]));
  }

  template <class D, class W, class T>
  void Distribution<D,W,T>::MLE(MA::ConstArray<D> const& data,
      std::vector<size_t> const& indexes) {
    assert(data.size().size() > 1);
    assert(data.size()[0] > 0);
    std::vector<W> weights(data.size()[0], 1);
    MLE(data, MA::ConstArray<W>({data.size()[0]}, &weights[0]), indexes);
  }

  template <class D, class W, class T>
  std::vector<size_t> Distribution<D,W,T>::sort_data(
      MA::ConstArray<D> const& data) {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);

    size_t counter = 0;

    std::vector<size_t> sorted_indexes(data.size()[0]);
    std::generate_n(sorted_indexes.begin(), data.size()[0],
        [&]() { return counter++; });

    D const* ptr = data.get_pointer();

    std::sort(sorted_indexes.begin(), sorted_indexes.end(),
        [&](size_t i, size_t j) { return ptr[i] < ptr[j]; });

    return sorted_indexes;
  }

  template <class D, class W, class T>
  D Distribution<D,W,T>::get_percentile(T p, MA::ConstArray<D> const& data,
      MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes) {
    return get_percentile(p, data, weight, indexes,
        build_percentile_vector(weight, indexes));
  }

  template <class D, class W, class T>
  D Distribution<D,W,T>::get_percentile(T p, MA::ConstArray<D> const& data,
      MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes,
      std::vector<T> const& percentile_vector) {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
    assert(data.size()[0] == indexes.size());

    auto data_size = data.size()[0];
    D const* ptr = data.get_pointer();

    std::vector<T> const& p_n = percentile_vector;

    if (p_n[0] >= p)
      return ptr[indexes[0]];
    else if (p_n[p_n.size()-1] <= p)
      return ptr[indexes[p_n.size()-1]];
    else {
      size_t start = 1, end = data_size-1;
      size_t mid;
      while (start < end) {
        mid = (start + end)/2;
        if (p_n[mid] >= p)
          end = mid;
        else
          start = mid+1;
      }

      size_t j = start;
      assert(p_n[j] >= p);
      assert(p_n[j-1] < p);
      T scale = (p - p_n[j-1])/(p_n[j] - p_n[j-1]);
      T offset = ptr[indexes[j-1]];
      return offset + scale * (ptr[indexes[j]] - ptr[indexes[j-1]]);
    }

    assert(false);
    return NAN;
  }

  template <class D, class W, class T>
  std::vector<T> Distribution<D,W,T>::build_percentile_vector(
      MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes) {
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == indexes.size());

    auto weight_size = weight.size()[0];

    T total_sum = 0;
    std::vector<T> p_n(weight_size);
    for (size_t j = 0; j < weight_size; j++) {
      T w = weight(indexes[j]);
      total_sum += w;
      p_n[j] = total_sum - w/2;
    }

    for (size_t j = 0; j < weight_size; j++)
      p_n[j] /= total_sum;

    return p_n;
  }
};

#endif
