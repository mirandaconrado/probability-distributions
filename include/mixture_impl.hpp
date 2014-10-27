#ifndef __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__

#include "mixture.hpp"

namespace ProbabilityDistributions {
  template <unsigned int K, class D, class W, class T, class... Dists>
  Mixture<K,D,W,T,Dists...>::Mixture(Dists&&... dists):
    stop_condition_(1e-4),
    max_iterations_(1000),
    mixture_weights_(sizeof...(Dists)),
    components_(tuple_type(dists...)) {
      create_component_pointers(
          CompileUtils::tuple_sequence_generator<tuple_type>());
    }

  template <unsigned int K, class D, class W, class T, class... Dists>
  template <size_t... S>
  void Mixture<K,D,W,T,Dists...>::create_component_pointers(
      CompileUtils::sequence<S...>) {
    components_pointers_ =
      std::vector<Distribution<D,W,T>*>({&std::get<S>(components_)...});
  }

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

  template <unsigned int K, class D, class W, class T, class... Dists>
  T Mixture<K,D,W,T,Dists...>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    MA::Array<W> expected_weight;
    build_expectation(expected_weight, weight, data);

    MA::Slice<W> weight_slice(expected_weight, 0);

    T ll = 0;
    for (unsigned int k = 0; k < K; k++)
      ll += components_pointers_[k]->log_likelihood(data,
          weight_slice.get_element(k));

    MA::Array<W> expected_weight_transp = transpose(expected_weight);

    ll += mixture_weights_(expected_weight_transp);
    return ll;
  }

  template <unsigned int K, class D, class W, class T, class... Dists>
  void Mixture<K,D,W,T,Dists...>::build_expectation(
      MA::Array<W>& expected_weight, MA::ConstArray<W> const& weight,
      MA::ConstArray<D> const& data) const {
    expected_weight.resize({K, weight.size()[0]});

    MA::ConstSlice<D> data_slice(data, 0);

    for (unsigned int i = 0; i < K; i++) {
      W sum = 0;
      for (unsigned int j = 0; j < data_slice.total_left_size(); j++) {
        expected_weight(i,j) = mixture_weights_.get_p()[i] *
          std::exp(components_pointers_[i]->log_likelihood(data_slice.get_element(j)));
        sum += expected_weight(i,j);
      }
      for (unsigned int j = 0; j < data_slice.total_left_size(); j++)
        expected_weight(i,j) *= weight(j) / sum;
    }
  }

  template <unsigned int K, class D, class W, class T, class... Dists>
  MA::Array<W> Mixture<K,D,W,T,Dists...>::transpose(
      MA::Array<W> const& original) const {
    MA::Array<W> ret({original.size()[1], original.size()[0]});

    for (MA::Size::SizeType::value_type i = 0; i < original.size()[0]; i++)
      for (MA::Size::SizeType::value_type j = 0; j < original.size()[1]; j++)
        ret(j,i) = original(i,j);

    return ret;
  }

  template <unsigned int K, class D, class W, class T, class... Dists>
  T Mixture<K,D,W,T,Dists...>::log_likelihood(MA::ConstArray<D> const& data,
      std::vector<unsigned int> const& labels) const {
    assert(labels.size() = data.size()[0]);
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == K);

    T ll = 0;

    MA::ConstSlice<D> data_slice(data, 0);
    for (size_t i = 0; i < data_slice.total_left_size(); i++) {
      unsigned int label = labels[i];
      ll += components_pointers_[label]->log_likelihood(data);
      ll += std::log(mixture_weights_->get_p()[label]);
    }

    return ll;
  }
};

#endif
