/*
* PatchMatchCUDA.cpp
*
* Copyright (c) 2014-2021 SEACAVE
*
* Author(s):
*
*	  cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*	  You are required to preserve legal notices and author attributions in
*	  that material or in the Appropriate Legal Notices displayed by works
*	  containing it.
*/

#include "Common.h"
#include "PatchMatchCUDA.h"
#include "DepthMap.h"

#ifdef _USE_CUDA

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

PatchMatchCUDA::PatchMatchCUDA(int device)
	: depthNormalEstimates(NULL)
	, hostLowDepths(NULL)
	, hostCostMap(NULL)
	, hostViewsMap(NULL)
	, stream(NULL)
	, cudaCameras(NULL)
	, cudaTextureImages(NULL)
	, cudaTextureDepths(NULL)
	, cudaDepthNormalEstimates(NULL)
	, cudaLowDepths(NULL)
	, cudaDepthNormalCosts(NULL)
	, cudaRandStates(NULL)
	, cudaSelectedViews(NULL)
	, imageCapacity(0)
	, depthTextureCapacity(0)
	, pixelCapacity(0)
	, lowDepthCapacity(0)
{
	// initialize CUDA device if needed
	if (CUDA::devices.IsEmpty() && CUDA::initDevice(device) != CUDA_SUCCESS)
		return;
	EnsureCUDAStream();
}

PatchMatchCUDA::~PatchMatchCUDA()
{
	Release();
}

void PatchMatchCUDA::Release()
{
	if (stream)
		CUDA::checkCudaCall(cudaStreamSynchronize(stream));
	FOREACH(i, cudaImageArrays) {
		if (textureImages[i])
			cudaDestroyTextureObject(textureImages[i]);
		if (cudaImageArrays[i])
			cudaFreeArray(cudaImageArrays[i]);
	}
	cudaImageArrays.clear();
	textureImages.clear();

	FOREACH(i, cudaDepthArrays) {
		if (textureDepths[i])
			cudaDestroyTextureObject(textureDepths[i]);
		if (cudaDepthArrays[i])
			cudaFreeArray(cudaDepthArrays[i]);
	}
	cudaDepthArrays.clear();
	textureDepths.clear();

	for (float* buffer: imageUploadBuffers)
		if (buffer)
			cudaFreeHost(buffer);
	for (cudaEvent_t event: imageUploadEvents)
		if (event)
			cudaEventDestroy(event);
	imageUploadBuffers.clear();
	imageUploadCapacities.clear();
	imageUploadEvents.clear();
	imageUploadsPending.clear();
	for (float* buffer: depthUploadBuffers)
		if (buffer)
			cudaFreeHost(buffer);
	for (cudaEvent_t event: depthUploadEvents)
		if (event)
			cudaEventDestroy(event);
	depthUploadBuffers.clear();
	depthUploadCapacities.clear();
	depthUploadEvents.clear();
	depthUploadsPending.clear();

	images.clear();
	cameras.clear();

	ReleaseCUDA();
}

void PatchMatchCUDA::EnsureCUDAStream()
{
	if (CUDA::devices.IsEmpty())
		return;
	if (stream == NULL)
		CUDA::checkCudaCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void PatchMatchCUDA::ReleaseCUDA()
{
	if (stream)
		CUDA::checkCudaCall(cudaStreamSynchronize(stream));
	if (cudaTextureImages)
		cudaFree(cudaTextureImages);
	if (cudaCameras)
		cudaFree(cudaCameras);
	if (cudaDepthNormalEstimates)
		cudaFree(cudaDepthNormalEstimates);
	if (cudaDepthNormalCosts)
		cudaFree(cudaDepthNormalCosts);
	if (cudaRandStates)
		cudaFree(cudaRandStates);
	if (cudaSelectedViews)
		cudaFree(cudaSelectedViews);
	if (cudaTextureDepths)
		cudaFree(cudaTextureDepths);
	if (cudaLowDepths)
		cudaFree(cudaLowDepths);
	if (depthNormalEstimates)
		cudaFreeHost(depthNormalEstimates);
	if (hostLowDepths)
		cudaFreeHost(hostLowDepths);
	if (hostCostMap)
		cudaFreeHost(hostCostMap);
	if (hostViewsMap)
		cudaFreeHost(hostViewsMap);
	if (stream)
		cudaStreamDestroy(stream);

	depthNormalEstimates = NULL;
	hostLowDepths = NULL;
	hostCostMap = NULL;
	hostViewsMap = NULL;
	stream = NULL;
	cudaCameras = NULL;
	cudaTextureImages = NULL;
	cudaTextureDepths = NULL;
	cudaDepthNormalEstimates = NULL;
	cudaLowDepths = NULL;
	cudaDepthNormalCosts = NULL;
	cudaRandStates = NULL;
	cudaSelectedViews = NULL;
	imageCapacity = 0;
	depthTextureCapacity = 0;
	pixelCapacity = 0;
	lowDepthCapacity = 0;
}

void PatchMatchCUDA::Init(bool bGeomConsistency)
{
	if (CUDA::devices.IsEmpty())
		return;
	EnsureCUDAStream();
	if (bGeomConsistency) {
		params.bGeomConsistency = true;
		params.nEstimationIters = 1;
	} else {
		params.bGeomConsistency = false;
		params.nEstimationIters = OPTDENSE::nEstimationIters;
	}
}

void PatchMatchCUDA::EnsurePatchMatchCUDA(size_t numImages, size_t numPixels)
{
	if (numImages > imageCapacity) {
		if (cudaTextureImages)
			CUDA::checkCudaCall(cudaFree(cudaTextureImages));
		if (cudaCameras)
			CUDA::checkCudaCall(cudaFree(cudaCameras));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * numImages));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaCameras, sizeof(Camera) * numImages));
		imageCapacity = numImages;
	}

	const size_t numDepthTextures(numImages > 0 ? numImages-1 : 0);
	if (params.bGeomConsistency && numDepthTextures > depthTextureCapacity) {
		if (cudaTextureDepths)
			CUDA::checkCudaCall(cudaFree(cudaTextureDepths));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * numDepthTextures));
		depthTextureCapacity = numDepthTextures;
	}

	if (numPixels > pixelCapacity) {
		if (cudaDepthNormalEstimates)
			CUDA::checkCudaCall(cudaFree(cudaDepthNormalEstimates));
		if (cudaDepthNormalCosts)
			CUDA::checkCudaCall(cudaFree(cudaDepthNormalCosts));
		if (cudaSelectedViews)
			CUDA::checkCudaCall(cudaFree(cudaSelectedViews));
		if (cudaRandStates)
			CUDA::checkCudaCall(cudaFree(cudaRandStates));
		if (depthNormalEstimates)
			CUDA::checkCudaCall(cudaFreeHost(depthNormalEstimates));
		if (hostCostMap)
			CUDA::checkCudaCall(cudaFreeHost(hostCostMap));
		if (hostViewsMap)
			CUDA::checkCudaCall(cudaFreeHost(hostViewsMap));

		CUDA::checkCudaCall(cudaMallocHost((void**)&depthNormalEstimates, sizeof(Point4) * numPixels));
		CUDA::checkCudaCall(cudaMallocHost((void**)&hostCostMap, sizeof(float) * numPixels));
		CUDA::checkCudaCall(cudaMallocHost((void**)&hostViewsMap, sizeof(uint32_t) * numPixels));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaDepthNormalEstimates, sizeof(Point4) * numPixels));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaDepthNormalCosts, sizeof(float) * numPixels));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaSelectedViews, sizeof(unsigned) * numPixels));
		CUDA::checkCudaCall(cudaMalloc((void**)&cudaRandStates, sizeof(curandState) * numPixels));
		pixelCapacity = numPixels;
	}
}

void PatchMatchCUDA::EnsureLowDepthCUDA(size_t numPixels)
{
	if (numPixels <= lowDepthCapacity)
		return;
	if (cudaLowDepths)
		CUDA::checkCudaCall(cudaFree(cudaLowDepths));
	if (hostLowDepths)
		CUDA::checkCudaCall(cudaFreeHost(hostLowDepths));
	CUDA::checkCudaCall(cudaMalloc((void**)&cudaLowDepths, sizeof(float) * numPixels));
	CUDA::checkCudaCall(cudaMallocHost((void**)&hostLowDepths, sizeof(float) * numPixels));
	lowDepthCapacity = numPixels;
}

float* PatchMatchCUDA::EnsureImageUploadBuffer(size_t index, size_t numPixels)
{
	const size_t numUploadBuffers(2);
	if (imageUploadBuffers.empty()) {
		imageUploadBuffers.resize(numUploadBuffers, NULL);
		imageUploadCapacities.resize(numUploadBuffers, 0);
		imageUploadEvents.resize(numUploadBuffers, NULL);
		imageUploadsPending.resize(numUploadBuffers, false);
		for (cudaEvent_t& event: imageUploadEvents)
			CUDA::checkCudaCall(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
	}
	const size_t slot(index % numUploadBuffers);
	if (imageUploadsPending[slot]) {
		CUDA::checkCudaCall(cudaEventSynchronize(imageUploadEvents[slot]));
		imageUploadsPending[slot] = false;
	}
	if (imageUploadCapacities[slot] < numPixels) {
		if (imageUploadBuffers[slot])
			CUDA::checkCudaCall(cudaFreeHost(imageUploadBuffers[slot]));
		CUDA::checkCudaCall(cudaMallocHost((void**)&imageUploadBuffers[slot], sizeof(float) * numPixels));
		imageUploadCapacities[slot] = numPixels;
	}
	return imageUploadBuffers[slot];
}

float* PatchMatchCUDA::EnsureDepthUploadBuffer(size_t index, size_t numPixels)
{
	const size_t numUploadBuffers(2);
	if (depthUploadBuffers.empty()) {
		depthUploadBuffers.resize(numUploadBuffers, NULL);
		depthUploadCapacities.resize(numUploadBuffers, 0);
		depthUploadEvents.resize(numUploadBuffers, NULL);
		depthUploadsPending.resize(numUploadBuffers, false);
		for (cudaEvent_t& event: depthUploadEvents)
			CUDA::checkCudaCall(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
	}
	const size_t slot(index % numUploadBuffers);
	if (depthUploadsPending[slot]) {
		CUDA::checkCudaCall(cudaEventSynchronize(depthUploadEvents[slot]));
		depthUploadsPending[slot] = false;
	}
	if (depthUploadCapacities[slot] < numPixels) {
		if (depthUploadBuffers[slot])
			CUDA::checkCudaCall(cudaFreeHost(depthUploadBuffers[slot]));
		CUDA::checkCudaCall(cudaMallocHost((void**)&depthUploadBuffers[slot], sizeof(float) * numPixels));
		depthUploadCapacities[slot] = numPixels;
	}
	return depthUploadBuffers[slot];
}

void PatchMatchCUDA::RecordImageUpload(size_t index)
{
	const size_t slot(index % imageUploadBuffers.size());
	CUDA::checkCudaCall(cudaEventRecord(imageUploadEvents[slot], stream));
	imageUploadsPending[slot] = true;
}

void PatchMatchCUDA::RecordDepthUpload(size_t index)
{
	const size_t slot(index % depthUploadBuffers.size());
	CUDA::checkCudaCall(cudaEventRecord(depthUploadEvents[slot], stream));
	depthUploadsPending[slot] = true;
}

void PatchMatchCUDA::AllocateImageCUDA(size_t i, const cv::Mat1f& image, bool bInitImage, bool bInitDepthMap)
{
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	if (bInitImage) {
		CUDA::checkCudaCall(cudaMallocArray(&cudaImageArrays[i], &channelDesc, image.cols, image.rows));

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cudaImageArrays[i];

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode  = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA::checkCudaCall(cudaCreateTextureObject(&textureImages[i], &resDesc, &texDesc, NULL));
	}

	if (params.bGeomConsistency && i > 0) {
		if (!bInitDepthMap) {
			textureDepths[i-1] = 0;
			cudaDepthArrays[i-1] = NULL;
			return;
		}

		CUDA::checkCudaCall(cudaMallocArray(&cudaDepthArrays[i-1], &channelDesc, image.cols, image.rows));

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cudaDepthArrays[i-1];

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode  = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA::checkCudaCall(cudaCreateTextureObject(&textureDepths[i-1], &resDesc, &texDesc, NULL));
	}
}

void PatchMatchCUDA::EstimateDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.size() > 1);

	// multi-resolution
	DepthData& fullResDepthData(depthData);
	const unsigned totalScaleNumber(params.bGeomConsistency ? 0u : OPTDENSE::nSubResolutionLevels);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	ViewsMap lowResViewsMap;
	IIndex prevNumImages = (IIndex)images.size();
	const IIndex numImages = depthData.images.size();
	params.nNumViews = (int)numImages-1;
	params.nInitTopK = std::min(params.nInitTopK, params.nNumViews);
	params.fDepthMin = depthData.dMin;
	params.fDepthMax = depthData.dMax;
	const size_t maxNumPixels(depthData.images.front().image.size().area());
	EnsurePatchMatchCUDA(numImages, maxNumPixels);
	if (totalScaleNumber > 0)
		EnsureLowDepthCUDA(maxNumPixels);
	if (prevNumImages < numImages) {
		images.resize(numImages);
		cameras.resize(numImages);
		cudaImageArrays.resize(numImages);
		textureImages.resize(numImages);
	}
	if (params.bGeomConsistency && cudaDepthArrays.size() < (size_t)params.nNumViews) {
		cudaDepthArrays.resize(params.nNumViews);
		textureDepths.resize(params.nNumViews);
	}
	const int maxPixelViews(MINF(params.nNumViews, 4));
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		const float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(DepthMapsData::ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		const Image8U::Size size(depthData.images.front().image.size());
		params.bLowResProcessed = false;
		if (scaleNumber != totalScaleNumber) {
			// all resolutions, but the smallest one, if multi-resolution is enabled
			params.bLowResProcessed = true;
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0, cv::INTER_LINEAR);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			cv::resize(lowResViewsMap, depthData.viewsMap, size, 0, 0, cv::INTER_NEAREST);
		} else {
			if (totalScaleNumber > 0) {
				// smallest resolution, when multi-resolution is enabled
				fullResDepthData.depthMap.release();
				fullResDepthData.normalMap.release();
				fullResDepthData.confMap.release();
				fullResDepthData.viewsMap.release();
			}
			// smallest resolution if multi-resolution is enabled; highest otherwise
			if (depthData.viewsMap.empty())
				depthData.viewsMap.create(size);
		}
		if (scaleNumber == 0) {
			// highest resolution
			if (depthData.confMap.empty())
				depthData.confMap.create(size);
		}

		// set keep threshold to:
		params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep;
		if (totalScaleNumber) {
			// multi-resolution enabled
			if (scaleNumber > 0 && scaleNumber != totalScaleNumber) {
				// all sub-resolutions, but the smallest and highest
				params.fThresholdKeepCost = 0.f; // disable filtering
			} else if (scaleNumber == totalScaleNumber || (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters)) {
				// smallest sub-resolution OR highest resolution and geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		} else {
			// multi-resolution disabled
			if (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters) {
				// geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		}

		for (IIndex i = 0; i < numImages; ++i) {
			const DepthData::ViewData& view = depthData.images[i];
			Image32F image = view.image;
			Camera camera;
			camera.K = Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.K.val).cast<float>();
			camera.R = Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.R.val).cast<float>();
			camera.C = Eigen::Map<const SEACAVE::Point3::EVec>(view.camera.C.ptr()).cast<float>();
			camera.height = image.rows;
			camera.width = image.cols;
			if (i >= prevNumImages) {
				// allocate image CUDA memory
				AllocateImageCUDA(i, image, true, !view.depthMap.empty());
			} else
			if (images[i].size() != image.size()) {
				// reallocate image CUDA memory
				cudaDestroyTextureObject(textureImages[i]);
				cudaFreeArray(cudaImageArrays[i]);
				if (params.bGeomConsistency && i > 0) {
					cudaDestroyTextureObject(textureDepths[i-1]);
					cudaFreeArray(cudaDepthArrays[i-1]);
				}
				AllocateImageCUDA(i, image, true, !view.depthMap.empty());
			} else
			if (params.bGeomConsistency && i > 0 && (view.depthMap.empty() != (cudaDepthArrays[i-1] == NULL))) {
				// reallocate depth CUDA memory
				if (cudaDepthArrays[i-1]) {
					cudaDestroyTextureObject(textureDepths[i-1]);
					cudaFreeArray(cudaDepthArrays[i-1]);
				}
				AllocateImageCUDA(i, image, false, !view.depthMap.empty());
			}
			const size_t rowSize(sizeof(float) * image.cols);
			float* imageUpload = EnsureImageUploadBuffer(i, image.size().area());
			for (int r=0; r<image.rows; ++r)
				memcpy(imageUpload+(size_t)r*image.cols, image.ptr<float>(r), rowSize);
			CUDA::checkCudaCall(cudaMemcpy2DToArrayAsync(cudaImageArrays[i], 0, 0, imageUpload, rowSize, rowSize, image.rows, cudaMemcpyHostToDevice, stream));
			RecordImageUpload(i);
			if (params.bGeomConsistency && i > 0 && !view.depthMap.empty()) {
				// set previously computed depth-map
				DepthMap depthMap(view.depthMap);
				if (depthMap.size() != image.size())
					cv::resize(depthMap, depthMap, image.size(), 0, 0, cv::INTER_LINEAR);
				const size_t depthRowSize(sizeof(float) * depthMap.cols);
				float* depthUpload = EnsureDepthUploadBuffer(i-1, depthMap.size().area());
				for (int r=0; r<depthMap.rows; ++r)
					memcpy(depthUpload+(size_t)r*depthMap.cols, depthMap.ptr<float>(r), depthRowSize);
				CUDA::checkCudaCall(cudaMemcpy2DToArrayAsync(cudaDepthArrays[i-1], 0, 0, depthUpload, depthRowSize, depthRowSize, depthMap.rows, cudaMemcpyHostToDevice, stream));
				RecordDepthUpload(i-1);
			}

			images[i] = std::move(image);
			cameras[i] = std::move(camera);
		}
		if (params.bGeomConsistency && cudaDepthArrays.size() > numImages - 1) {
			for (IIndex i = numImages; i < prevNumImages; ++i) {
				// free image CUDA memory
				cudaDestroyTextureObject(textureDepths[i-1]);
				cudaFreeArray(cudaDepthArrays[i-1]);
			}
			cudaDepthArrays.resize(params.nNumViews);
			textureDepths.resize(params.nNumViews);
		}
		if (prevNumImages > numImages) {
			for (IIndex i = numImages; i < prevNumImages; ++i) {
				// free image CUDA memory
				cudaDestroyTextureObject(textureImages[i]);
				cudaFreeArray(cudaImageArrays[i]);
			}
			images.resize(numImages);
			cameras.resize(numImages);
			cudaImageArrays.resize(numImages);
			textureImages.resize(numImages);
		}
		prevNumImages = numImages;

		// setup CUDA memory
		CUDA::checkCudaCall(cudaMemcpyAsync(cudaTextureImages, textureImages.data(), sizeof(cudaTextureObject_t) * numImages, cudaMemcpyHostToDevice, stream));
		CUDA::checkCudaCall(cudaMemcpyAsync(cudaCameras, cameras.data(), sizeof(Camera) * numImages, cudaMemcpyHostToDevice, stream));
		if (params.bGeomConsistency) {
			// set previously computed depth-maps
			ASSERT(depthData.depthMap.size() == depthData.GetView().image.size());
			CUDA::checkCudaCall(cudaMemcpyAsync(cudaTextureDepths, textureDepths.data(), sizeof(cudaTextureObject_t) * params.nNumViews, cudaMemcpyHostToDevice, stream));
		}

		// load depth-map and normal-map into CUDA memory
		for (int r = 0; r < depthData.depthMap.rows; ++r) {
			const int baseIndex = r * depthData.depthMap.cols;
			for (int c = 0; c < depthData.depthMap.cols; ++c) {
				const Normal& n = depthData.normalMap(r, c);
				const int index = baseIndex + c;
				Point4& depthNormal = depthNormalEstimates[index];
				depthNormal.topLeftCorner<3, 1>() = Eigen::Map<const Normal::EVec>(n.ptr());
				depthNormal.w() = depthData.depthMap(r, c);
			}
		}
		CUDA::checkCudaCall(cudaMemcpyAsync(cudaDepthNormalEstimates, depthNormalEstimates, sizeof(Point4) * depthData.depthMap.size().area(), cudaMemcpyHostToDevice, stream));

		// load low resolution depth-map into CUDA memory
		if (params.bLowResProcessed) {
			ASSERT(depthData.depthMap.isContinuous());
			const size_t numPixels(depthData.depthMap.size().area());
			memcpy(hostLowDepths, depthData.depthMap.ptr<float>(), sizeof(float) * numPixels);
			CUDA::checkCudaCall(cudaMemcpyAsync(cudaLowDepths, hostLowDepths, sizeof(float) * numPixels, cudaMemcpyHostToDevice, stream));
		}

		// run CUDA patch-match
		ASSERT(!depthData.viewsMap.empty());
		RunCUDA(depthData.confMap.getData(), (uint32_t*)depthData.viewsMap.getData());
		CUDA::checkCudaCall(cudaGetLastError());

		// load depth-map, normal-map and confidence-map from CUDA memory
		for (int r = 0; r < depthData.depthMap.rows; ++r) {
			for (int c = 0; c < depthData.depthMap.cols; ++c) {
				const int index = r * depthData.depthMap.cols + c;
				const Point4& depthNormal = depthNormalEstimates[index];
				const Depth depth = depthNormal.w();
				ASSERT(std::isfinite(depth));
				depthData.depthMap(r, c) = depth;
				depthData.normalMap(r, c) = depthNormal.topLeftCorner<3, 1>();
				if (scaleNumber == 0) {
					// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
					ASSERT(!depthData.confMap.empty());
					float& conf = depthData.confMap(r, c);
					conf = conf >= 1.f ? 0.f : 1.f - conf;
					// map pixel views from bit-mask to index
					ASSERT(!depthData.viewsMap.empty());
					ViewsID& views = depthData.viewsMap(r, c);
					if (depth > 0) {
						const uint32_t bitviews(*reinterpret_cast<const uint32_t*>(views.val));
						int j = 0;
						for (int i = 0; i < 32; ++i) {
							if (bitviews & (1 << i)) {
								views[j] = i;
								if (++j == maxPixelViews)
									break;
							}
						}
						while (j < 4)
							views[j++] = 255;
					} else
						views = ViewsID(255, 255, 255, 255);
				}
			}
		}
		
		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
			lowResViewsMap = depthData.viewsMap;
		}

	}

	// apply ignore mask
	if (OPTDENSE::nIgnoreMaskLabel >= 0) {
		const DepthData::ViewData& view = depthData.GetView();
		BitMatrix mask;
		if (DepthEstimator::ImportIgnoreMask(*view.pImageData, depthData.depthMap.size(), (uint16_t)OPTDENSE::nIgnoreMaskLabel, mask))
			depthData.ApplyIgnoreMask(mask);
	}

	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.GetSize() > 2 ?
		String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
		String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		images.front().cols, images.front().rows, TD_TIMER_GET_FMT().c_str());
}
/*----------------------------------------------------------------*/

#endif // _USE_CUDA
