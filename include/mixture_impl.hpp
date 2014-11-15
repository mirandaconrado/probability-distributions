#ifndef __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__MIXTURE_IMPL_HPP__

#include "mixture.hpp"

#include "log_number.hpp"

namespace ProbabilityDistributions {
  template <unsigned int SS, class D, class W, class T, class... Dists>
  Mixture<SS,D,W,T,Dists...>::Mixture(Dists&&... dists):
    have_asymmetric_(CompileUtils::or_<
        std::is_base_of<
          AsymmetricDistribution<typename CompileUtils::clean_type<Dists>::type,D,W,T>,
          typename CompileUtils::clean_type<Dists>::type>::value...>::value),
    stop_condition_(1e-4),
    max_iterations_(1000),
    mixture_weights_(),
    components_(tuple_type(dists...)) {
      create_component_pointers(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <size_t... S>
  void Mixture<SS,D,W,T,Dists...>::create_component_pointers(
      CompileUtils::sequence<S...>) {
    components_pointers_ =
      std::vector<Distribution<D,W,T>*>({&std::get<S>(components_)...});
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <class RNG>
  void Mixture<SS,D,W,T,Dists...>::sample(MA::Array<D>& samples,
      size_t n_samples, RNG& rng) const {
    MA::Size::SizeType size(2);
    size[0] = n_samples;
    size[1] = SS;
    samples.resize(size);

    MA::Array<D> components_samples;
    mixture_weights_.sample(components_samples, n_samples, rng);
    MA::Array<unsigned int> components_indexes;
    mixture_weights_.sample_to_index(components_indexes, components_samples);

    MA::Slice<D> slice(samples, 0);
    for (size_t i = 0; i < n_samples; i++) {
      MA::Array<D> sample = slice.get_element(i);
      sample_components(sample, components_indexes(i), rng,
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <class RNG, size_t... S>
  void Mixture<SS,D,W,T,Dists...>::sample_components(MA::Array<D>& sample,
      size_t component_id, RNG& rng, CompileUtils::sequence<S...>) const {
    bool vec[] = {sample_component<S>(sample, component_id, rng)...};
    (void)vec;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  template <size_t I, class RNG>
  bool Mixture<SS,D,W,T,Dists...>::sample_component(MA::Array<D>& sample,
      size_t component_id, RNG& rng) const {
    if (I != component_id)
      return false;
    std::get<I>(components_).sample(sample, 1, rng);
    return true;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  T Mixture<SS,D,W,T,Dists...>::log_likelihood(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight) const {
    check_data_and_weight(data, weight);

    MA::ConstSlice<D> data_slice(data, 0);

    MA::Array<W> null_weight({1});
    null_weight(0) = 1;
    MA::Size sample_size({1, SS});

    T ll = 0;
    for (unsigned int j = 0; j < data_slice.total_left_size(); j++) {
      MA::ConstArray<D> sample = data_slice.get_element(j);
      sample.resize(sample_size);

      T local_likelihood = 0;
      for (unsigned int i = 0; i < K; i++)
        local_likelihood += mixture_weights_.get_p()[i] *
          std::exp(components_pointers_[i]->log_likelihood(sample, null_weight));

      ll += weight(j) * std::log(local_likelihood);
    }

    return ll;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void Mixture<SS,D,W,T,Dists...>::build_expectation(
      MA::Array<W>& expected_weight, MA::ConstArray<W> const& weight,
      MA::ConstArray<D> const& data) const {
    expected_weight.resize({K, weight.size()[0]});

    MA::ConstSlice<D> data_slice(data, 0);

    MA::Array<W> null_weight({1});
    null_weight(0) = 1;
    MA::Size sample_size({1, SS});

    for (unsigned int j = 0; j < data_slice.total_left_size(); j++) {
      MA::ConstArray<D> sample = data_slice.get_element(j);
      sample.resize(sample_size);

      LogNumber sum = 0;
      std::vector<LogNumber> vals(K);
      for (unsigned int i = 0; i < K; i++) {
        vals[i].from_log(
          components_pointers_[i]->log_likelihood(sample, null_weight));
        vals[i] *= mixture_weights_.get_p()[i];
        sum += vals[i];
      }

      for (unsigned int i = 0; i < K; i++)
        expected_weight(i,j) = weight(j) * (vals[i] / sum).to_double();
    }
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  MA::Array<W> Mixture<SS,D,W,T,Dists...>::transpose(
      MA::Array<W> const& original) const {
    MA::Array<W> ret({original.size()[1], original.size()[0]});

    for (MA::Size::SizeType::value_type i = 0; i < original.size()[0]; i++)
      for (MA::Size::SizeType::value_type j = 0; j < original.size()[1]; j++)
        ret(j,i) = original(i,j);

    return ret;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  T Mixture<SS,D,W,T,Dists...>::log_likelihood(MA::ConstArray<D> const& data,
      std::vector<unsigned int> const& labels) const {
    assert(labels.size() = data.size()[0]);
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == SS);

    T ll = 0;

    MA::ConstSlice<D> data_slice(data, 0);
    for (size_t i = 0; i < data_slice.total_left_size(); i++) {
      unsigned int label = labels[i];
      ll += components_pointers_[label]->log_likelihood(data);
      ll += std::log(mixture_weights_->get_p()[label]);
    }

    return ll;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void Mixture<SS,D,W,T,Dists...>::MLE(MA::ConstArray<D> const& data,
      MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes) {
    check_data_and_weight(data, weight);

    std::vector<size_t> const* index_pointer = &indexes;

    for (unsigned int i = 0; i < K; i++)
      if (components_pointers_[i]->require_sorted() && indexes.size() == 0) {
        index_pointer = new
          std::vector<size_t>(Distribution<D,W,T>::sort_data(data));
        break;
      }

    MA::Array<W> expected_weight;

    bool fixed_asymmetries = false;
    if (have_asymmetric_) {
      fixed_asymmetries = true;
      fix_all_asymmetries(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

    while (1) {
      T ll_old = -INFINITY, ll_new = -INFINITY;
      size_t it = 0;

      do {
        build_expectation(expected_weight, weight, data);

        ll_old = ll_new;

        MA::Slice<W> weight_slice(expected_weight, 0);

        for (unsigned int k = 0; k < K; k++)
          components_pointers_[k]->MLE(data, weight_slice.get_element(k),
              *index_pointer);

        MA::Array<W> expected_weight_transp = transpose(expected_weight);

        mixture_weights_.MLE(expected_weight_transp);

        ll_new = log_likelihood(data, weight);
        //assert(ll_new >= ll_old);
        assert(ll_new >= ll_old - 1e-8);

        it++;
      } while(ll_new - ll_old > stop_condition_ && it < max_iterations_);

      if (!fixed_asymmetries)
        break;

      fixed_asymmetries = false;
      free_all_asymmetries(
          typename CompileUtils::tuple_sequence_generator<tuple_type>::type());
    }

    if (index_pointer != &indexes)
      delete index_pointer;
  }

  template <unsigned int SS, class D, class W, class T, class... Dists>
  void Mixture<SS,D,W,T,Dists...>::check_data_and_weight(
      MA::ConstArray<D> const& data, MA::ConstArray<W> const& weight) const {
    assert(data.size().size() == 2);
    assert(data.size()[0] > 0);
    assert(data.size()[1] == SS);
    assert(weight.size().size() == 1);
    assert(weight.size()[0] == data.size()[0]);
  }
};

#endif
