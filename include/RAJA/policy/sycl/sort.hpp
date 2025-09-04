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

#if defined(RAJA_ENABLE_SYCL)

namespace RAJA
{
namespace impl
{
namespace scan
{



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
#if 0
  detail::openmp::sort(detail::UnstableSorter {}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
#endif

  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);    
}

/*!
        \brief stable sort given range using comparison function
*/
template<typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Sycl>,
                      type_traits::is_sycl_policy<ExecPolicy>>
stable(resources::Sycl sycl_res,
       const ExecPolicy&,
       Iter begin,
       Iter end,
       Compare comp)
{
#if 0
  detail::openmp::sort(detail::StableSorter {}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
#endif

#warning "Nulled stable"

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
#if 0
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::openmp::sort(detail::UnstableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
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

#if 0
  // openmp example
  auto begin    = RAJA::zip(keys_begin, vals_begin);
  auto end      = RAJA::zip(keys_end, vals_begin + (keys_end - keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::openmp::sort(detail::StableSorter {}, begin, end,
                       RAJA::compare_first<zip_ref>(comp));
  return resources::EventProxy<resources::Host>(host_res);
#endif
  
  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
}

}  // namespace scan
}  // namespace impl
}  // namespace RAJA
  


#endif  // closing endif for RAJA enable Sycl guard

#endif  // closing endif for header include guard

