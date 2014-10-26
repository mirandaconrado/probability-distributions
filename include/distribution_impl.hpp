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

    std::vector<size_t> sorted_indexes(data.size()[0]);
    std::generate_n(sorted_indexes.begin(), data.size()[0],
        []() { static size_t counter = 0; return counter++; });

    std::sort(sorted_indexes.begin(), sorted_indexes.end(),
        [&](size_t i, size_t j) { return data(i,0) <  data(j,0); });

    return sorted_indexes;
  }
};

#endif
