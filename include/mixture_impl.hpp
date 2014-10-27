#ifndef __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__

#include "mixture.hpp"

namespace ProbabilityDistributions {
  template <unsigned int K, class D, class W, class T, class... Dists>
  Mixture<K,D,W,T,Dists...>::Mixture(Dists&&... dists):
    stop_condition_(1e-4),
    max_iterations_(1000),
    mixture_weights_(sizeof...(Dists)),
    components_(tuple_type(dists...)) { }

  template <unsigned int K, class D, class W, class T, class... Dists>
  template <class RNG>
  void Mixture<K,D,W,T,Dists...>::sample(MA::Array<D>& samples,
      size_t n_samples, RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = K;
    samples.resize(size);

    MA::Array<D> components_samples;
    mixture_weights_.sample(components_samples, n_samples, rng);
    MA::Array<unsigned int> components_indexes;
    mixture_weights_.sample_to_index(components_indexes, components_samples);

    MA::Slice<D> slice(samples, 0);
    for (size_t i = 0; i < n_samples; i++) {
      MA::Array<D> sample = slice.get_element(i);
      sample_components(sample, components_indexes(i), rng,
          CompileUtils::tuple_sequence_generator<tuple_type>());
    }
  }

  template <unsigned int K, class D, class W, class T, class... Dists>
  template <class RNG, size_t... S>
  void Mixture<K,D,W,T,Dists...>::sample_components(MA::Array<D>& sample,
      size_t component_id, RNG& rng, CompileUtils::sequence<S...>) const {
    bool vec[] = {sample_component<S>(sample, component_id, rng)...};
  }

  template <unsigned int K, class D, class W, class T, class... Dists>
  template <size_t I, class RNG>
  bool Mixture<K,D,W,T,Dists...>::sample_component(MA::Array<D>& sample,
      size_t component_id, RNG& rng) const {
    if (I != component_id)
      return false;
    std::get<I>(components_).sample(sample, 1, rng);
  }
};

#endif
