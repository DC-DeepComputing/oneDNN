/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef TEST_AARCH64_MMLA_COMMON_HPP
#define TEST_AARCH64_MMLA_COMMON_HPP

#include "oneapi/dnnl/dnnl.hpp"

#if DNNL_AARCH64
#include "src/cpu/aarch64/cpu_isa_traits.hpp"

namespace dnnl {
namespace aarch64_mmla_test {

inline bool has_bf16_mmla_impl() {
    using namespace impl::cpu::aarch64;
    const auto isa = dnnl_get_effective_cpu_isa();
    return mayiuse_bf16()
            && (isa == cpu_isa_traits<sve_128>::user_option_val
                    || isa == cpu_isa_traits<sve_256>::user_option_val);
}

inline bool matches_weights_desc(
        const memory::desc &desc, int k_dim, int n_dim) {
    if (desc.get_data_type() != memory::data_type::bf16
            || desc.get_format_kind() != memory::format_kind::blocked)
        return false;

    // BF16 MMLA packs each K8 block as K2 x N x K4, where N is one
    // ISA-sized output-channel tile.
    const auto inner_blks = desc.get_inner_blks();
    const auto inner_idxs = desc.get_inner_idxs();
    return inner_idxs == memory::dims {k_dim, n_dim, k_dim}
    && (inner_blks == memory::dims {2, 16, 4}
            || inner_blks == memory::dims {2, 32, 4});
}

} // namespace aarch64_mmla_test
} // namespace dnnl
#endif

#endif
