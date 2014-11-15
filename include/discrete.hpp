#ifndef __PROBABILITY_DISTRIBUTIONS__DISCRETE_HPP__
#define __PROBABILITY_DISTRIBUTIONS__DISCRETE_HPP__

#include "distribution.hpp"

namespace ProbabilityDistributions {
  template <unsigned int K, class D, class W = D, class T = W>
  class Discrete: public Distribution<D,W,T> {
    public:
      static_assert(K > 0, "Can't create distribution without classes.");

      static constexpr unsigned int sample_size = K;

      Discrete();
      explicit Discrete(std::vector<T> const& p);

      void set_p(std::vector<T> const& p) { assert(p.size() == K); p_ = p; normalize(); }
      std::vector<T> const& get_p() const { return p_; }
      unsigned int get_number_of_classes() const { return K; }
      Discrete<K,D,W,T> const&
        operator=(Discrete<K,D,W,T> const& other) {
          set_p(other.get_p());
          return *this;
        }

      template <class RNG>
      void sample(MA::Array<D>& samples, size_t n_samples, RNG& rng) const;

      using Distribution<D,W,T>::log_likelihood;
      T log_likelihood(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;

      using Distribution<D,W,T>::MLE;
      void MLE(MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight,
          std::vector<size_t> const& indexes = std::vector<size_t>());

      void sample_to_index(MA::Array<unsigned int>& indexes,
          MA::ConstArray<D> const& samples) const;

      void index_to_sample(MA::Array<D>& samples,
          MA::ConstArray<unsigned int> const& indexes) const;

    private:
      void check_data_and_weight(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight) const;
      void normalize();

      std::vector<T> p_;
  };
};

#include "discrete_impl.hpp"

#endif
