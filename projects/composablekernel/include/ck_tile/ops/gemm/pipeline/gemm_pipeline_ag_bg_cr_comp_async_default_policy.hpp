// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    // For FP8/BF8: 4, 12, 16
    template <typename DataType, index_t KPack>
    static constexpr bool IsSupportedAsyncVectorWidth =
        sizeof(DataType) * KPack == 4 || sizeof(DataType) * KPack == 12 ||
        sizeof(DataType) * KPack == 16;

    // XOR Swizzle: support FP8 / BF8
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleDataType =
        (std::is_same_v<remove_cvref_t<typename Problem::ADataType>, fp8_t> ||  // A FP8
         std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf8_t>) && // A BF8
        (std::is_same_v<remove_cvref_t<typename Problem::BDataType>, fp8_t> ||  // B FP8
         std::is_same_v<remove_cvref_t<typename Problem::BDataType>, bf8_t>);   // B BF8

    // Check that async vector store to LDS is supported
    template <typename Problem>
    static constexpr bool IsSupportedXorSwizzleAsyncWidth =
        IsSupportedAsyncVectorWidth<typename Problem::ADataType, GetSmemPackA<Problem>()> &&
        IsSupportedAsyncVectorWidth<typename Problem::BDataType, GetSmemPackB<Problem>()>;

    // Assume normal LDS layout, not transpose-load
    template <typename Problem>
    static constexpr bool UseXorSwizzle =
        !is_a_load_tr<Problem> && !is_b_load_tr<Problem> &&
        IsSupportedXorSwizzleDataType<Problem> && IsSupportedXorSwizzleAsyncWidth<Problem>;

    // Get number of LDS read accesses by warp GEMM
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWGAttrNumAccess()
    {
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();

        if constexpr(!(is_a_load_tr<Problem> || is_b_load_tr<Problem>))
        {
            // Current XOR-swizzle
            return WGAttrNumAccessEnum::Single;
        }
        else if constexpr(vector_size == thread_elements)
        {
            // 1 ds_read_tr instruction needed
            return WGAttrNumAccessEnum::Single;
        }
        else if constexpr(vector_size * 2 == thread_elements)
        {
            // 2 ds_read_tr instructions needed
            return WGAttrNumAccessEnum::Double;
        }
        else if constexpr(vector_size * 4 == thread_elements)
        {
            // 4 ds_read_tr instructions needed
            return WGAttrNumAccessEnum::Quad;
        }
        else
        {
            // Unsupported ds_read_tr access pattern
            return WGAttrNumAccessEnum::Invalid;
        }
    }

    template <typename Problem, index_t MNPerBlock, index_t WarpTileMN, index_t K2>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXorSwizzledABLdsBlockDescriptor()
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t KWarps    = BlockWarps::at(I2);
        constexpr index_t K1        = WarpTile::at(I2) / K2;
        constexpr index_t K0        = KPerBlock / (KWarps * K1 * K2);

        constexpr index_t warp_size            = get_warp_size();
        constexpr index_t warp_num             = BlockSize / warp_size;
        constexpr auto wg_attr_num_access      = GetWGAttrNumAccess<Problem>();
        constexpr index_t wg_attr_num_access_v = static_cast<index_t>(wg_attr_num_access);

        static_assert(warp_num * warp_size == BlockSize, "Wrong!");
        static_assert(KWarps * K0 * K1 * K2 == KPerBlock, "Wrong!");
        static_assert(wg_attr_num_access == WGAttrNumAccessEnum::Single,
                      "XOR swizzle currently supports only the normal non-ds_read_tr path");

        constexpr index_t M4 = warp_size / wg_attr_num_access_v / K1;
        constexpr index_t M3 = wg_attr_num_access_v;
        constexpr index_t M2 = WarpTileMN / M4 / M3;
        constexpr index_t M1 = (warp_num / Problem::NumWaveGroups) / M2;
        constexpr index_t M0 = MNPerBlock / M1 / M2 / M3 / M4;

        static_assert(M1 * M0 * M2 * M3 * M4 == MNPerBlock, "Wrong!");

        constexpr auto desc_0 =
            make_naive_tensor_descriptor(number_tuple<M2, KWarps, M1, M0, K0, M3, M4, K1, K2>{},
                                         number_tuple<KWarps * M1 * M0 * K0 * M3 * M4 * K1 * K2,
                                                      M1 * M0 * K0 * M3 * M4 * K1 * K2,
                                                      M0 * K0 * M3 * M4 * K1 * K2,
                                                      K0 * M3 * M4 * K1 * K2,
                                                      M3 * M4 * K1 * K2,
                                                      M4 * K1 * K2,
                                                      K1 * K2,
                                                      K2,
                                                      1>{},
                                         number<K2>{},
                                         number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M2>{}),
                       make_pass_through_transform(number<KWarps>{}),
                       make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M3>{}),
                       make_xor_transform(make_tuple(number<M4>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            container_concat(generate_tuple([](auto i) { return sequence<i>{}; }, number<6>{}),
                             make_tuple(sequence<6, 7>{}),
                             make_tuple(sequence<8>{})),
            container_concat(generate_tuple([](auto i) { return sequence<i>{}; }, number<6>{}),
                             make_tuple(sequence<6, 7>{}),
                             make_tuple(sequence<8>{})));

        constexpr auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M3, M4>{}),
                       make_merge_transform_v3_division_mod(number_tuple<KWarps, K0, K1, K2>{})),
            make_tuple(sequence<3, 2, 0, 5, 6>{}, sequence<1, 4, 7, 8>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_2;
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeALdsBlockDescriptor
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            if constexpr(UseXorSwizzle<Problem>)
            {
                using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
                constexpr index_t KPack = GetSmemPackA<Problem>();
                constexpr auto desc     = MakeXorSwizzledABLdsBlockDescriptor<Problem,
                                                                              MPerBlock,
                                                                              WarpTile::at(I0),
                                                                              KPack>();
                static_assert(desc.get_element_space_size() == MPerBlock * KPerBlock,
                              "XOR swizzle must not change A LDS allocation size");
                return desc;
            }
            else
            {
                constexpr index_t KPack = GetSmemPackA<Problem>();

                constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                return transform_tensor_descriptor(
                    a_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<MPerBlock>{}),
                               make_merge_transform(
                                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeBLdsBlockDescriptor
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            if constexpr(UseXorSwizzle<Problem>)
            {
                using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
                constexpr index_t KPack = GetSmemPackB<Problem>();
                constexpr auto desc     = MakeXorSwizzledABLdsBlockDescriptor<Problem,
                                                                              NPerBlock,
                                                                              WarpTile::at(I1),
                                                                              KPack>();
                static_assert(desc.get_element_space_size() == NPerBlock * KPerBlock,
                              "XOR swizzle must not change B LDS allocation size");
                return desc;
            }
            else
            {
                constexpr index_t KPack = GetSmemPackB<Problem>();

                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<KPerBlock / KPack>{}, number<NPerBlock>{}, number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});

                return transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(make_pass_through_transform(number<NPerBlock>{}),
                               make_merge_transform(
                                   make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <typename Problem, index_t K2, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadABDramWindow(const Window& window)
    {
        using BlockGemmShape = typename Problem::BlockGemmShape;
        using BlockWarps     = typename BlockGemmShape::BlockWarps;
        using WarpTile       = typename BlockGemmShape::WarpTile;

        constexpr auto ndims = std::decay_t<decltype(window)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");

        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr index_t KWarps    = BlockWarps::at(I2);
        constexpr index_t K1        = WarpTile::at(I2) / K2;

        static_assert(K1 * K2 == WarpTile::at(I2), "Wrong!");
        static_assert(KPerBlock % (KWarps * K1 * K2) == 0, "Wrong!");
        static_assert(GetWGAttrNumAccess<Problem>() == WGAttrNumAccessEnum::Single,
                      "XOR swizzle currently supports only the normal non-ds_read_tr path");

        constexpr index_t M4 = get_warp_size() / K1;
        static_assert(M4 * K1 == get_warp_size(), "Wrong!");

        auto&& tensor_view      = window.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view.get_tensor_descriptor().get_lengths();

        const index_t k_tiles = cols / (KWarps * K1 * K2);
        const auto col_lens   = make_tuple(k_tiles, number<KWarps>{}, number<K1>{}, number<K2>{});

        const index_t M0    = integer_divide_ceil(rows, M4);
        const auto row_lens = make_tuple(M0, number<M4>{});

        const auto desc_0 = transform_tensor_descriptor(
            tensor_view.get_tensor_descriptor(),
            make_tuple(make_unmerge_transform(row_lens), make_unmerge_transform(col_lens)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}));

        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M4>{}, number<K1>{})),
                       make_pass_through_transform(k_tiles),
                       make_pass_through_transform(number<KWarps>{}),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1, 4>{}, sequence<2>{}, sequence<3>{}, sequence<5>{}));

        const auto desc =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4, 5>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tile_window(
            make_tensor_view<address_space_enum::global>(&tensor_view.get_buffer_view()(0), desc),
            window.get_window_lengths(),
            window.get_window_origin());
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const Window& window)
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t KPack = GetSmemPackA<Problem>();
            return MakeAsyncLoadABDramWindow<Problem, KPack>(window);
        }
        else
        {
            return make_tile_window(window.get_bottom_tensor_view(),
                                    window.get_window_lengths(),
                                    window.get_window_origin());
        }
    }

    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadBDramWindow(const Window& window)
    {
        if constexpr(UseXorSwizzle<Problem>)
        {
            constexpr index_t KPack = GetSmemPackB<Problem>();
            return MakeAsyncLoadABDramWindow<Problem, KPack>(window);
        }
        else
        {
            return make_tile_window(window.get_bottom_tensor_view(),
                                    window.get_window_lengths(),
                                    window.get_window_origin());
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr auto wg_attr_num_access = GetWGAttrNumAccess<Problem>();

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
