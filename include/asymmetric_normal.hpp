#ifndef __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_NORMAL_HPP__
#define __PROBABILITY_DISTRIBUTIONS__ASYMMETRIC_NORMAL_HPP__

#include "asymmetric_distribution.hpp"

#include <boost/random/normal_distribution.hpp>

namespace ProbabilityDistributions {
  template <class D, class W = D, class T = W>
  class AsymmetricNormal:
    public AsymmetricDistribution<AsymmetricNormal<D,W,T>,D,W,T> {
    public:
      AsymmetricNormal(T p, T mu, T sigma);

      static constexpr unsigned int sample_size = 1;

      void fix_sigma(bool fixed = true) { fixed_sigma_ = fixed; }
      void set_sigma(T sigma) {
        assert(sigma > 0); sigma_ = sigma; sigma2_ = 2*sigma*sigma; }
      T get_sigma() const { return sigma_; }
      AsymmetricNormal<D,W,T> const&
        operator=(AsymmetricNormal<D,W,T> const& other) {
          base_class::set_p(other.get_p());
          base_class::set_mu(other.get_mu());
          set_sigma(other.get_sigma());
          base_class::fixed_p_ = other.fixed_p_;
          base_class::fixed_mu_ = other.fixed_mu_;
          fixed_sigma_ = other.fixed_sigma_;
          return *this;
        }

    private:
      typedef AsymmetricDistribution<AsymmetricNormal<D,W,T>,D,W,T> base_class;
      friend class AsymmetricDistribution<AsymmetricNormal<D,W,T>,D,W,T>;

      struct TruncatedNormal {
        TruncatedNormal(T sigma): dist(0, sigma) { }

        template <class RNG>
        T operator()(RNG& rng) { return std::abs(dist(rng)); }

        boost::random::normal_distribution<T> dist;
      };

      TruncatedNormal create_gamma_plus() const;
      TruncatedNormal create_gamma_minus() const;

      T constant_likelihood() const;
      T negative_ll(T s, T mu) const;
      T positive_ll(T s, T mu) const;
      void set_parameter_vector(std::vector<T> const& p) {
        assert(p.size() == 1); sigma_ = p[0]; }
      std::vector<T> get_parameter_vector() const { return {sigma_}; }

      void init_MLE(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);
      void end_MLE();
      void updated_p();

      void MLE_fixed_p(MA::ConstArray<D> const& data,
          MA::ConstArray<W> const& weight, std::vector<size_t> const& indexes);

      bool fixed_sigma_;
      T sigma_, sigma2_, alpha_, alpha_inv_, alpha2_, alpha2_inv_;
      std::vector<T> pos_sum_all_0_, pos_sum_all_1_, pos_sum_all_2_;
      std::vector<T> neg_sum_all_0_, neg_sum_all_1_, neg_sum_all_2_;
  };
};

#include "asymmetric_normal_impl.hpp"

#endif
