////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////
#include <new>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include "MaskedOcclusionCulling.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler specific functions: MSC and Intel compilers supported and there is rudimentary support for clang/llvm with SSE4.1 only
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __clang__

#ifdef __SSE4_1__

	#include <immintrin.h>

    #include <mm_malloc.h>

	#define FORCE_INLINE inline

	static FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
	{
		unsigned long idx;
		idx = __builtin_ctzl(*mask);
		*mask &= *mask - 1;
		return idx;
	}

	static void *aligned_alloc(size_t alignment, size_t size)
	{
		return _mm_malloc(size, alignment);
	}
	
	static void aligned_free(void *ptr)
	{
		_mm_free(ptr);
	}
	
	// Detect AVX2 / SSE4.1 support - in clang/llvm this is a compile flag, no runtime checking for supported vector instruction as of yet.
	static MaskedOcclusionCulling::Implementation GetCPUInstructionSet()
	{
		static MaskedOcclusionCulling::Implementation instructionSet = MaskedOcclusionCulling::SSE41;
		return instructionSet;
	}

#else
#error Only SSE4.1 codepath supported on clang at the moment
#endif

#else

#ifdef _MSC_VER

	#if defined(__AVX__) || defined(__AVX2__)
	// For performance reasons, the MaskedOcclusionCullingAVX2.cpp file should be compiled with VEX encoding for SSE instructions (to avoid 
	// AVX-SSE transition penalties, see https://software.intel.com/en-us/articles/avoiding-avx-sse-transition-penalties). However, the SSE
	// version in MaskedOcclusionCulling.cpp _must_ be compiled without VEX encoding to allow backwards compatibility. Best practice is to 
	// use lowest supported target platform (/arch:SSE2) as project default, and elevate only the MaskedOcclusionCullingAVX2.cpp file.
	#error The MaskedOcclusionCulling.cpp should be compiled with lowest supported target platform, e.g. /arch:SSE2
	#endif

	#include <intrin.h>

	#define FORCE_INLINE __forceinline

	static FORCE_INLINE unsigned long find_clear_lsb(unsigned int *mask)
	{
		unsigned long idx;
		_BitScanForward(&idx, *mask);
		*mask &= *mask - 1;
		return idx;
	}

	static void *aligned_alloc(size_t alignment, size_t size)
	{
		return _aligned_malloc(size, alignment);
	}
	
	static void aligned_free(void *ptr)
	{
		_aligned_free(ptr);
	}
	
	// Detect AVX2 / SSE4.1 support
	static MaskedOcclusionCulling::Implementation GetCPUInstructionSet()
	{
		static bool initialized = false;
		static MaskedOcclusionCulling::Implementation instructionSet = MaskedOcclusionCulling::SSE2;
		
		int cpui[4];
		if (!initialized)
		{
			initialized = true;
			instructionSet = MaskedOcclusionCulling::SSE2;
	
			int nIds, nExIds;
			__cpuid(cpui, 0);
			nIds = cpui[0];
			__cpuid(cpui, 0x80000000);
			nExIds = cpui[0];
	
			if (nIds >= 7 && nExIds >= 0x80000001)
			{
				// Test AVX2 support
				instructionSet = MaskedOcclusionCulling::AVX2;
				__cpuidex(cpui, 1, 0);
				if ((cpui[2] & 0x18401000) != 0x18401000)
					instructionSet = MaskedOcclusionCulling::SSE2;
				__cpuidex(cpui, 7, 0);
				if ((cpui[1] & 0x128) != 0x128)
					instructionSet = MaskedOcclusionCulling::SSE2;
				__cpuidex(cpui, 0x80000001, 0);
				if ((cpui[2] & 0x20) != 0x20)
					instructionSet = MaskedOcclusionCulling::SSE2;
				if (instructionSet == MaskedOcclusionCulling::AVX2 && (_xgetbv(0) & 0x6) != 0x6)
					instructionSet = MaskedOcclusionCulling::SSE2;
			}
			if (instructionSet == MaskedOcclusionCulling::SSE2 && nIds >= 1)
			{
				// Test SSE4.1 support
				instructionSet = MaskedOcclusionCulling::SSE41;
				__cpuidex(cpui, 1, 0);
				if ((cpui[2] & 0x080000) != 0x080000)
					instructionSet = MaskedOcclusionCulling::SSE2;
			}
		}
		return instructionSet;
	}

#endif // #ifdef _MSC_VER

#endif // #ifdef __clang__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compiler compatibility helpers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __clang__

// Wrap the native type so we can use our operators on it without fear that we'll mix it up with compiler-generated operators 
// that do something slightly different (like with clang/llvm)
typedef union __declspec( align( 16 ) ) moc__m128 {
	__m128              native;
	float               m128_f32[4];
	uint64_t			m128_u64[2];
	int8_t              m128_i8[16];
	int16_t             m128_i16[8];
	int32_t             m128_i32[4];
	int64_t             m128_i64[2];
	uint8_t				m128_u8[16];
	uint16_t			m128_u16[8];
	uint32_t			m128_u32[4];
	moc__m128( ) = default;
	explicit moc__m128( const __m128 & ref ) { native = ref; }
	operator __m128( void ) const { return native; }
} moc__m128;

typedef union __declspec( align( 16 ) ) moc__m128i {
	__m128i             native;
	int8_t              m128i_i8[16];
	int16_t             m128i_i16[8];
	int32_t             m128i_i32[4];
	int64_t             m128i_i64[2];
	uint8_t				m128i_u8[16];
	uint16_t			m128i_u16[8];
	uint32_t			m128i_u32[4];
	uint64_t			m128i_u64[2];
	moc__m128i( ) = default;
	explicit moc__m128i( const __m128i & ref ) { native = ref; }
	operator __m128i(void) const { return native; }
} moc__m128i;

#else // #ifdef __clang__

typedef __m128          moc__m128;
typedef __m128i         moc__m128i;

#endif // #ifdef __clang__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions (not directly related to the algorithm/rasterizer)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MaskedOcclusionCulling::TransformVertices(const float *mtx, const float *inVtx, float *xfVtx, unsigned int nVtx, const VertexLayout &vtxLayout)
{
	// This function pretty slow, about 10-20% slower than if the vertices are stored in aligned SOA form.
	if (nVtx == 0)
		return;

	// Load matrix and swizzle out the z component. For post-multiplication (OGL), the matrix is assumed to be column 
	// major, with one column per SSE register. For pre-multiplication (DX), the matrix is assumed to be row major.
	moc__m128 mtxCol0 = (moc__m128)_mm_loadu_ps(mtx);
	moc__m128 mtxCol1 = (moc__m128)_mm_loadu_ps(mtx + 4);
	moc__m128 mtxCol2 = (moc__m128)_mm_loadu_ps(mtx + 8);
	moc__m128 mtxCol3 = (moc__m128)_mm_loadu_ps(mtx + 12);

	int stride = vtxLayout.mStride;
	const char *vPtr = (const char *)inVtx;
	float *outPtr = xfVtx;

	// Iterate through all vertices and transform
	for (unsigned int vtx = 0; vtx < nVtx; ++vtx)
	{
		moc__m128 xVal = (moc__m128)_mm_load1_ps((float*)(vPtr));
		moc__m128 yVal = (moc__m128)_mm_load1_ps((float*)(vPtr + vtxLayout.mOffsetY));
		moc__m128 zVal = (moc__m128)_mm_load1_ps((float*)(vPtr + vtxLayout.mOffsetZ));

		moc__m128 xform = (moc__m128)_mm_add_ps(_mm_mul_ps(mtxCol0, xVal), _mm_add_ps(_mm_mul_ps(mtxCol1, yVal), _mm_add_ps(_mm_mul_ps(mtxCol2, zVal), mtxCol3)));
		_mm_storeu_ps(outPtr, xform);
		vPtr += stride;
		outPtr += 4;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Typedefs
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef MaskedOcclusionCulling::pfnAlignedAlloc pfnAlignedAlloc;
typedef MaskedOcclusionCulling::pfnAlignedFree  pfnAlignedFree;
typedef MaskedOcclusionCulling::VertexLayout    VertexLayout;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 defines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SIMD_LANES             4
#define TILE_HEIGHT_SHIFT      2

#define SIMD_LANE_IDX moc__m128i(_mm_setr_epi32(0, 1, 2, 3))

#define SIMD_SUB_TILE_COL_OFFSET moc__m128i(_mm_setr_epi32(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3))
#define SIMD_SUB_TILE_ROW_OFFSET moc__m128i(_mm_setzero_si128())
#define SIMD_SUB_TILE_COL_OFFSET_F moc__m128(_mm_setr_ps(0, SUB_TILE_WIDTH, SUB_TILE_WIDTH * 2, SUB_TILE_WIDTH * 3))
#define SIMD_SUB_TILE_ROW_OFFSET_F moc__m128(_mm_setzero_ps())

#define SIMD_LANE_YCOORD_I moc__m128i(_mm_setr_epi32(128, 384, 640, 896))
#define SIMD_LANE_YCOORD_F moc__m128(_mm_setr_ps(128.0f, 384.0f, 640.0f, 896.0f))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Common SSE2/SSE4.1 functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef moc__m128 __mw;
typedef moc__m128i __mwi;

#define mw_f32 m128_f32
#define mw_i32 m128i_i32

#define _mmw_set1_ps (__mw)_mm_set1_ps
#define _mmw_setzero_ps (__mw)_mm_setzero_ps
#define _mmw_andnot_ps (__mw)_mm_andnot_ps
#define _mmw_min_ps (__mw)_mm_min_ps
#define _mmw_max_ps (__mw)_mm_max_ps
#define _mmw_movemask_ps _mm_movemask_ps
#define _mmw_cmpge_ps(a,b) (__mw)_mm_cmpge_ps(a, b)
#define _mmw_cmpgt_ps(a,b) (__mw)_mm_cmpgt_ps(a, b)
#define _mmw_cmpeq_ps(a,b) (__mw)_mm_cmpeq_ps(a, b)
#define _mmw_fmadd_ps(a,b,c) (__mw)_mm_add_ps(_mm_mul_ps(a,b), c)
#define _mmw_fmsub_ps(a,b,c) (__mw)_mm_sub_ps(_mm_mul_ps(a,b), c)
#define _mmw_shuffle_ps (__mw)_mm_shuffle_ps
#define _mmw_insertf32x4_ps(a,b,c) (b)

#define _mmw_set1_epi32 (__mwi)_mm_set1_epi32
#define _mmw_setzero_epi32 (__mwi)_mm_setzero_si128
#define _mmw_andnot_epi32 (__mwi)_mm_andnot_si128
#define _mmw_subs_epu16 (__mwi)_mm_subs_epu16
#define _mmw_cmpeq_epi32 (__mwi)_mm_cmpeq_epi32
#define _mmw_cmpgt_epi32 (__mwi)_mm_cmpgt_epi32
#define _mmw_srai_epi32 (__mwi)_mm_srai_epi32
#define _mmw_srli_epi32 (__mwi)_mm_srli_epi32
#define _mmw_slli_epi32 (__mwi)_mm_slli_epi32
#define _mmw_abs_epi32 (__mwi)_mm_abs_epi32

#define _mmw_cvtps_epi32 (__mwi)_mm_cvtps_epi32
#define _mmw_cvttps_epi32 (__mwi)_mm_cvttps_epi32
#define _mmw_cvtepi32_ps (__mw)_mm_cvtepi32_ps

#define _mmx_fmadd_ps _mmw_fmadd_ps
#define _mmx_max_epi32 _mmw_max_epi32
#define _mmx_min_epi32 _mmw_min_epi32

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialized SSE input assembly function for general vertex gather 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FORCE_INLINE void GatherVertices(moc__m128 *vtxX, moc__m128 *vtxY, moc__m128 *vtxW, const float *inVtx, const unsigned int *inTrisPtr, int numLanes, const VertexLayout &vtxLayout)
{
	for (int lane = 0; lane < numLanes; lane++)
	{
		for (int i = 0; i < 3; i++)
		{
			char *vPtrX = (char *)inVtx + inTrisPtr[lane * 3 + i] * vtxLayout.mStride;
			char *vPtrY = vPtrX + vtxLayout.mOffsetY;
			char *vPtrW = vPtrX + vtxLayout.mOffsetW;

			vtxX[i].m128_f32[lane] = *((float*)vPtrX);
			vtxY[i].m128_f32[lane] = *((float*)vPtrY);
			vtxW[i].m128_f32[lane] = *((float*)vPtrW);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE4.1 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE41
{
	static FORCE_INLINE moc__m128i _mmw_mullo_epi32(const moc__m128i &a, const moc__m128i &b) { return (moc__m128i)_mm_mullo_epi32(a, b); }
	static FORCE_INLINE moc__m128i _mmw_min_epi32(const moc__m128i &a, const moc__m128i &b) { return (moc__m128i)_mm_min_epi32(a, b); }
	static FORCE_INLINE moc__m128i _mmw_max_epi32(const moc__m128i &a, const moc__m128i &b) { return (moc__m128i)_mm_max_epi32(a, b); }
	static FORCE_INLINE moc__m128 _mmw_blendv_ps(const moc__m128 &a, const moc__m128 &b, const moc__m128 &c) { return (moc__m128)_mm_blendv_ps(a, b, c); }
#if PRECISE_COVERAGE != 0
	static FORCE_INLINE moc__m128i _mmw_blendv_epi32(const moc__m128i &a, const moc__m128i &b, const moc__m128i &c)
	{
		return (moc__m128i)_mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _mm_castsi128_ps(c)));
	}
//  unused at the moment
//	static FORCE_INLINE moc__m128i _mmw_blendv_epi32(const moc__m128i &a, const moc__m128i &b, const moc__m128 &c)
//	{
//		return (moc__m128i)_mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), c));
//	}
#endif
	static FORCE_INLINE int _mmw_testz_epi32(const moc__m128i &a, const moc__m128i &b) { return _mm_testz_si128(a, b); }
	static FORCE_INLINE moc__m128 _mmx_dp4_ps(const moc__m128 &a, const moc__m128 &b) { return (moc__m128)_mm_dp_ps(a, b, 0xFF); }
#if PRECISE_COVERAGE == 0
	static FORCE_INLINE moc__m128 _mmw_floor_ps(const moc__m128 &a) { return (moc__m128)_mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
	static FORCE_INLINE moc__m128 _mmw_ceil_ps(const moc__m128 &a) { return (moc__m128)_mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);	}
#endif
	static FORCE_INLINE moc__m128i _mmw_transpose_epi8(const moc__m128i &a)
	{
		const moc__m128i shuff = (moc__m128i)_mm_setr_epi8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
		return (moc__m128i)_mm_shuffle_epi8(a, shuff);
	}
	static FORCE_INLINE moc__m128i _mmw_sllv_ones(const moc__m128i &ishift)
	{
		moc__m128i shift = (moc__m128i)_mm_min_epi32(ishift, _mm_set1_epi32(32));

		// Uses lookup tables and _mm_shuffle_epi8 to perform _mm_sllv_epi32(~0, shift)
		const moc__m128i byteShiftLUT = (moc__m128i)_mm_setr_epi8((char)(~0u), (char)(~0u << 1), (char)(~0u << 2), (char)(~0u << 3), (char)(~0u << 4), (char)(~0u << 5), (char)(~0u << 6), (char)(~0u << 7), 0, 0, 0, 0, 0, 0, 0, 0);
		const moc__m128i byteShiftOffset = (moc__m128i)_mm_setr_epi8(0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24);
		const moc__m128i byteShiftShuffle = (moc__m128i)_mm_setr_epi8(0x0, 0x0, 0x0, 0x0, 0x4, 0x4, 0x4, 0x4, 0x8, 0x8, 0x8, 0x8, 0xC, 0xC, 0xC, 0xC);

		moc__m128i byteShift = (moc__m128i)_mm_shuffle_epi8(shift, byteShiftShuffle);
		byteShift = (moc__m128i)_mm_min_epi8(_mm_subs_epu8(byteShift, byteShiftOffset), _mm_set1_epi8(8));
		moc__m128i retMask = (moc__m128i)_mm_shuffle_epi8(byteShiftLUT, byteShift);

		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE41;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

#ifndef __clang__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SSE2 version
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace MaskedOcclusionCullingSSE2
{
	static FORCE_INLINE moc__m128i _mmw_mullo_epi32(const moc__m128i &a, const moc__m128i &b) 
	{ 
		// Do products for even / odd lanes & merge the result
		moc__m128i even = (moc__m128i)_mm_and_si128(_mm_mul_epu32(a, b), _mm_setr_epi32(~0, 0, ~0, 0));
		moc__m128i odd = (moc__m128i)_mm_slli_epi64(_mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32)), 32);
		return (moc__m128i)_mm_or_si128(even, odd);
	}
	static FORCE_INLINE moc__m128i _mmw_min_epi32(const moc__m128i &a, const moc__m128i &b) 
	{ 
		moc__m128i cond = (moc__m128i)_mm_cmpgt_epi32(a, b);
		return (moc__m128i)_mm_or_si128(_mm_andnot_si128(cond, a), _mm_and_si128(cond, b));
	}
	static FORCE_INLINE moc__m128i _mmw_max_epi32(const moc__m128i &a, const moc__m128i &b) 
	{ 
		moc__m128i cond = (moc__m128i)_mm_cmpgt_epi32(b, a);
		return (moc__m128i)_mm_or_si128(_mm_andnot_si128(cond, a), _mm_and_si128(cond, b));
	}
	static FORCE_INLINE int _mmw_testz_epi32(const moc__m128i &a, const moc__m128i &b) 
	{ 
		return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(a, b), _mm_setzero_si128())) == 0xFFFF;
	}
	static FORCE_INLINE moc__m128 _mmw_blendv_ps(const moc__m128 &a, const moc__m128 &b, const moc__m128 &c)
	{	
		moc__m128 cond = (moc__m128)_mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c), 31));
		return (moc__m128)_mm_or_ps(_mm_andnot_ps(cond, a), _mm_and_ps(cond, b));
	}
	static FORCE_INLINE moc__m128i _mmw_blendv_epi32(const moc__m128i &a, const moc__m128i &b, const moc__m128i &c)
	{
		return (moc__m128i)_mm_castps_si128(_mmw_blendv_ps((moc__m128)_mm_castsi128_ps(a), (moc__m128)_mm_castsi128_ps(b), (moc__m128)_mm_castsi128_ps(c)));
	}
	static FORCE_INLINE moc__m128i _mmw_blendv_epi32(const moc__m128i &a, const moc__m128i &b, const moc__m128 &c)
	{
		return (moc__m128i)_mm_castps_si128(_mmw_blendv_ps((moc__m128)_mm_castsi128_ps(a), (moc__m128)_mm_castsi128_ps(b), c));
	}
	static FORCE_INLINE moc__m128 _mmx_dp4_ps(const moc__m128 &a, const moc__m128 &b)
	{ 
		// Product and two shuffle/adds pairs (similar to hadd_ps)
		moc__m128 prod = (moc__m128)_mm_mul_ps(a, b);
		moc__m128 dp = (moc__m128)_mm_add_ps(prod, _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1)));
		dp = (moc__m128)_mm_add_ps(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(0, 1, 2, 3)));
		return dp;
	}
	static FORCE_INLINE moc__m128 _mmw_floor_ps(const moc__m128 &a) 
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
		moc__m128 rounded = (moc__m128)_mm_cvtepi32_ps(_mm_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE moc__m128 _mmw_ceil_ps(const moc__m128 &a) 
	{ 
		int originalMode = _MM_GET_ROUNDING_MODE();
		_MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
		moc__m128 rounded = (moc__m128)_mm_cvtepi32_ps(_mm_cvtps_epi32(a));
		_MM_SET_ROUNDING_MODE(originalMode);
		return rounded;
	}
	static FORCE_INLINE moc__m128i _mmw_transpose_epi8(const moc__m128i &a)
	{
		// Perform transpose through two 16->8 bit pack and byte shifts
		moc__m128i res = a;
		const moc__m128i mask = (moc__m128i)_mm_setr_epi8(~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0, ~0, 0);
		res = (moc__m128i)_mm_packus_epi16(_mm_and_si128(res, mask), _mm_srli_epi16(res, 8));
		res = (moc__m128i)_mm_packus_epi16(_mm_and_si128(res, mask), _mm_srli_epi16(res, 8));
		return res;
	}
	static FORCE_INLINE moc__m128i _mmw_sllv_ones(const moc__m128i &ishift)
	{
		moc__m128i shift = _mmw_min_epi32(ishift, (moc__m128i)_mm_set1_epi32(32));
		
		// Uses scalar approach to perform _mm_sllv_epi32(~0, shift)
		static const unsigned int maskLUT[33] = {
			~0U << 0, ~0U << 1, ~0U << 2 ,  ~0U << 3, ~0U << 4, ~0U << 5, ~0U << 6 , ~0U << 7, ~0U << 8, ~0U << 9, ~0U << 10 , ~0U << 11, ~0U << 12, ~0U << 13, ~0U << 14 , ~0U << 15,
			~0U << 16, ~0U << 17, ~0U << 18 , ~0U << 19, ~0U << 20, ~0U << 21, ~0U << 22 , ~0U << 23, ~0U << 24, ~0U << 25, ~0U << 26 , ~0U << 27, ~0U << 28, ~0U << 29, ~0U << 30 , ~0U << 31,
			0U };

		moc__m128i retMask;
		retMask.m128i_u32[0] = maskLUT[shift.m128i_u32[0]];
		retMask.m128i_u32[1] = maskLUT[shift.m128i_u32[1]];
		retMask.m128i_u32[2] = maskLUT[shift.m128i_u32[2]];
		retMask.m128i_u32[3] = maskLUT[shift.m128i_u32[3]];
		return retMask;
	}

	static MaskedOcclusionCulling::Implementation gInstructionSet = MaskedOcclusionCulling::SSE2;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Include common algorithm implementation (general, SIMD independent code)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#include "MaskedOcclusionCullingCommon.inl"

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Utility function to create a new object using the allocator callbacks
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
	{
		MaskedOcclusionCullingPrivate *object = (MaskedOcclusionCullingPrivate *)memAlloc(32, sizeof(MaskedOcclusionCullingPrivate));
		new (object) MaskedOcclusionCullingPrivate(memAlloc, memFree);
		return object;
	}
};

#endif // #ifndef __clang__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Object construction and allocation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __clang__ 
namespace MaskedOcclusionCullingAVX2
{
	extern MaskedOcclusionCulling *CreateMaskedOcclusionCulling(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree);
}
#endif // #ifndef __clang__

MaskedOcclusionCulling *MaskedOcclusionCulling::Create()
{
	return Create(aligned_alloc, aligned_free);
}

MaskedOcclusionCulling *MaskedOcclusionCulling::Create(pfnAlignedAlloc memAlloc, pfnAlignedFree memFree)
{
	MaskedOcclusionCulling *object = nullptr;
	Implementation instructionSet = GetCPUInstructionSet();

#ifndef __clang__
    if (instructionSet == AVX2)
		object = MaskedOcclusionCullingAVX2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use AVX2 optimized (fast) version
	else 
#endif // #ifndef __clang__
    if (instructionSet == SSE41)
		object = MaskedOcclusionCullingSSE41::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE4.1 version
#ifndef __clang__
    else
		object = MaskedOcclusionCullingSSE2::CreateMaskedOcclusionCulling(memAlloc, memFree);	// Use SSE2 (slow) version
#endif // #ifndef __clang__

    assert( object != nullptr );

	return object;
}

void MaskedOcclusionCulling::Destroy(MaskedOcclusionCulling *moc)
{
	pfnAlignedFree alignedFreeCallback = moc->mAlignedFreeCallback;
	moc->~MaskedOcclusionCulling();
	alignedFreeCallback(moc);
}
