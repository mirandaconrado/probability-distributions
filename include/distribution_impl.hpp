#ifndef __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISTRIBUTION_IMPL_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <class T>
  T Distribution<T>::log_likelihood(MA::ConstArray<T> const& data) const {
    assert(data.size().size() > 1);
    assert(data.size()[0] > 0);
    std::vector<T> weights(data.size()[0], 1);
    return log_likelihood(data, MA::ConstArray<T>({data.size()[0]},
          &weights[0]));
  }

  template <class T>
  void Distribution<T>::MLE(MA::ConstArray<T> const& data) {
    assert(data.size().size() > 1);
    assert(data.size()[0] > 0);
    std::vector<T> weights(data.size()[0], 1);
    MLE(data, MA::ConstArray<T>({data.size()[0]}, &weights[0]));
  }
};

#endif
