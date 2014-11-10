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

    std::sort(sorted_indexes.begin(), sorted_indexes.end(),
        [&](size_t i, size_t j) { return data(i,0) <  data(j,0); });

    return sorted_indexes;
  }

  template <class D, class W, class T>
  D Distribution<D,W,T>::get_percentile(T p, MA::ConstArray<D> const& data,
      MA::ConstArray<D> const& weight, std::vector<size_t> const& indexes) {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == 1);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
    assert(data.size()[0] == indexes.size());

    auto data_size = data.size()[0];
    D const* ptr = data.get_pointer();

    T total_sum = 0;
    std::vector<T> p_n(data_size);
    for (size_t j = 0; j < data_size; j++) {
      T w = weight(indexes[j]);
      total_sum += w;
      p_n[j] = total_sum - w/2;
    }

    for (size_t j = 0; j < data_size; j++)
      p_n[j] /= total_sum;

    if (p_n[0] >= p)
      return ptr[indexes[0]];
    else if (p_n[p_n.size()-1] <= p)
      return ptr[indexes[p_n.size()-1]];
    else {
      for (size_t j = 1; j < data_size; j++)
        if (p_n[j] >= p) {
          T scale = (p - p_n[j-1])/(p_n[j] - p_n[j-1]);
          T offset = ptr[indexes[j-1]];
          return offset + scale * (ptr[indexes[j]] - ptr[indexes[j-1]]);
        }
    }

    assert(false);
    return NAN;
  }
};

#endif
