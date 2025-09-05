/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_sycl_HPP
#define RAJA_sort_sycl_HPP

#include <algorithm>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#if defined(RAJA_ENABLE_SYCL)

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{
namespace sycl
{

// this number is arbitrary
constexpr int get_min_iterates_per_task() { return 128; }

#if 0

/*!
        \brief sort given range using sorter and comparison function
               by manually assigning work to threads
*/
template<typename Sorter, typename Iter, typename Compare>
inline void sort_parallel_region(Sorter sorter,
                                 Iter begin,
                                 RAJA::detail::IterDiff<Iter> n,
                                 Compare comp)
{
  using RAJA::detail::firstIndex;
  using diff_type = RAJA::detail::IterDiff<Iter>;

  const diff_type num_threads = omp_get_num_threads();

  const diff_type thread_id = omp_get_thread_num();

  const diff_type i_begin = firstIndex(n, num_threads, thread_id);
  {
    const diff_type i_end = firstIndex(n, num_threads, thread_id + 1);

    // this thread sorts range [i_begin, i_end)
    sorter(begin + i_begin, begin + i_end, comp);
  }

  // hierarchically merge ranges
  for (diff_type middle_offset = 1; middle_offset < num_threads;
       middle_offset *= 2)
  {

    diff_type end_offset = 2 * middle_offset;

    const diff_type i_middle = firstIndex(
        n, num_threads, std::min(thread_id + middle_offset, num_threads));
    const diff_type i_end = firstIndex(
        n, num_threads, std::min(thread_id + end_offset, num_threads));

#pragma omp barrier

    if (thread_id % end_offset == 0)
    {

      // this thread merges ranges [i_begin, i_middle) and [i_middle, i_end)
      // std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end,
      // comp);
      RAJA::detail::inplace_merge(begin + i_begin, begin + i_middle,
                                  begin + i_end, comp);
    }
  }
}

#endif

/*!
        \brief sort given range using sorter and comparison function
*/
template<typename Sorter, typename Iter, typename Compare>
inline void sort(resources::Sycl sycl_res, Sorter sorter, Iter begin, Iter end, Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;

  constexpr diff_type min_iterates_per_task = get_min_iterates_per_task();

  const diff_type n = end - begin;

  if (n <= min_iterates_per_task)
  {

    sorter(begin, end, comp);
  }
  else
  {
    ::sycl::queue* sycl_queue = sycl_res.get_queue();

    // Create a SYCL policy associated with the queue
    //::sycl::ext::oneapi::experimental::parallel_stl::sycl_execution_policy sycl_policy(q);

    //::sycl::ext::oneapi::dpl::execution::device_policy<> sycl_policy(q);

    auto policy = oneapi::dpl::execution::make_device_policy(*sycl_queue);

    // SGS comp
    // Sort the data using sycl::sort
    //::sycl::ext::oneapi::experimental::parallel_stl::sort(sycl_policy, begin, end);
    oneapi::dpl::sort(policy, begin, end, comp);

#if 0

    const diff_type max_threads = omp_get_max_threads();

    const diff_type requested_num_threads = std::min(
        (n + min_iterates_per_task - 1) / min_iterates_per_task, max_threads);
    RAJA_UNUSED_VAR(requested_num_threads);  // avoid warning in hip device code

#pragma omp parallel num_threads(static_cast <int>(requested_num_threads))
    {
      sort_parallel_region(sorter, begin, n, comp);
    }
#endif
  }
}

}  // namespace sycl

}  // namespace detail


/*!
        \brief sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Sycl>,
                      type_traits::is_sycl_policy<ExecPolicy>>
unstable(resources::Sycl sycl_res,
         const ExecPolicy&,
         Iter begin,
         Iter end,
         Compare comp)
{
  detail::sycl::sort(sycl_res, detail::UnstableSorter {}, begin, end, comp);

  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);    
}

/*!
        \brief stable sort given range using comparison function
*/

//
  
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Sycl>,
                      type_traits::is_sycl_policy<ExecPolicy>>
stable(resources::Sycl sycl_res,
       const ExecPolicy&,
       Iter begin,
       Iter end,
       Compare comp)
{
  detail::sycl::sort(sycl_res, detail::StableSorter {}, begin, end, comp);

  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);  
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template<typename ExecPolicy,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Sycl>,
                      type_traits::is_sycl_policy<ExecPolicy>>
unstable_pairs(resources::Sycl sycl_res,
               const ExecPolicy&,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               Compare comp)
{
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;

#if 0
  detail::sycl::sort(sycl_res, detail::UnstableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));
#endif

  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);  
}
  

/*!
        \brief stable sort given range of pairs using comparison function on
   keys
*/

// TODO why does SYCL scan not have enable_if_t logic?
template<typename ExecPolicy,
         typename KeyIter,
         typename ValIter,
         typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Sycl> ,
                      type_traits::is_sycl_policy<ExecPolicy>>
stable_pairs(resources::Sycl sycl_res,
             const ExecPolicy&,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             Compare comp)
{
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;

#if 0
  // SGS sycl not happy
  detail::sycl::sort(sycl_res, detail::StableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));
#endif
  
  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
}

}  // namespace sort
}  // namespace impl
}  // namespace RAJA
  


#endif  // closing endif for RAJA enable Sycl guard

#endif  // closing endif for header include guard

