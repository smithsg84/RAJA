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

//#define HAVE_ONEAPI_DPL
#ifdef HAVE_ONEAPI_DPL
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
  }
}
#else

template<typename Sorter, typename Iter, typename Compare>
inline void sort(resources::Sycl sycl_res, Sorter sorter, Iter begin, Iter end, Compare comp)
{
  using valueT = typename std::iterator_traits<Iter>::value_type;

  // Calculate the size of the input range
  size_t n = std::distance(begin, end);
  
  if (n <= 1) return;

  ::sycl::queue* sycl_queue = sycl_res.get_queue();

  // This does not work, but works in RAJA sycl scan?  
  // sycl::buffer<valueT, 1> tempAccBuff(begin, sycl::range<1>(n));
  
  // Create buffers for input and output data
  ::sycl::buffer<valueT, 1> input_buf(begin, end);
  ::sycl::buffer<valueT, 1> output_buf((::sycl::range<1>(n)));
        
  // Pointers to swap between input and output buffers
  ::sycl::buffer<valueT, 1>* current_buf = &input_buf;
  ::sycl::buffer<valueT, 1>* next_buf = &output_buf;

  // Bottom-up merge sort
  for (size_t width = 1; width < n; width *= 2) {
    size_t num_merges = (n + 2 * width - 1) / (2 * width);
    
    sycl_queue -> submit([&](::sycl::handler& h) {
      ::sycl::accessor current_acc(*current_buf, h, ::sycl::read_only);
      ::sycl::accessor next_acc(*next_buf, h, ::sycl::write_only);

      h.parallel_for(::sycl::range<1>(num_merges), [=](::sycl::id<1> idx) {
	size_t merge_idx = idx[0];
	size_t left_start = merge_idx * 2 * width;
	size_t left_end = std::min(left_start + width, n);
	size_t right_start = left_end;
	size_t right_end = std::min(left_start + 2 * width, n);
        
	// Merge two sorted subarrays
	size_t i = left_start, j = right_start, k = left_start;
        
	while (i < left_end && j < right_end) {
	  if (comp(current_acc[i], current_acc[j])) {
	    next_acc[k++] = current_acc[i++];
	  } else {
	    next_acc[k++] = current_acc[j++];
	  }
	}
        
	// Copy remaining elements
	while (i < left_end) {
	  next_acc[k++] = current_acc[i++];
	}
	while (j < right_end) {
	  next_acc[k++] = current_acc[j++];
	}
      });
    });
    
    // Swap buffers for next iteration
    std::swap(*current_buf, *next_buf);
  }
  
  // Wait for completion
  sycl_queue -> wait();
  
  // Copy result back to original vector
  ::sycl::host_accessor final_acc(*current_buf);
  auto it_data = begin;
  for (size_t i = 0; i < n; ++i) {
    *it_data = final_acc[i];
    it_data++;
  }
}

#endif
  

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




#ifdef LIKELY_NEED_THIS
// SGS sycl not happy, does not think the RAJA zip is device copyable
//

zip_tuple<true, double, long>
  
#include <sycl/sycl.hpp>

struct MyData {
    int a;
    float b;
};

namespace sycl {
template <>
struct is_device_copyable<MyData> : std::true_type {};
}

#endif


#if 0 
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

