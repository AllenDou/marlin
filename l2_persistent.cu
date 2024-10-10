#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

__global__ void reset_data(int* data_streaming, int const* lut_persistent,
                           size_t data_streaming_size,
                           size_t lut_persistent_size)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < data_streaming_size; i += stride)
    {
        data_streaming[i] = lut_persistent[i % lut_persistent_size];
    }
}

/**
 * @brief Reset the data_streaming using lut_persistent so that the
 * data_streaming is lut_persistent repeatedly.
 *
 * @param data_streaming The data for reseting.
 * @param lut_persistent The values for resetting data_streaming.
 * @param data_streaming_size The size for data_streaming.
 * @param lut_persistent_size The size for lut_persistent.
 * @param stream The CUDA stream.
 */
void launch_reset_data(int* data_streaming, int const* lut_persistent,
                       size_t data_streaming_size, size_t lut_persistent_size,
                       cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    reset_data<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        data_streaming, lut_persistent, data_streaming_size,
        lut_persistent_size);
    CHECK_LAST_CUDA_ERROR();
}

bool verify_data(int* data, int n, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data[i] != i % n)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    size_t num_megabytes_persistent_data{3};
    if (argc == 2)
    {
        num_megabytes_persistent_data = std::atoi(argv[1]);
    }

    constexpr int const num_repeats{100};
    constexpr int const num_warmups{10};

    cudaDeviceProp device_prop{};
    int current_device{0};
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, current_device));
    std::cout << "GPU: " << device_prop.name << std::endl;
    std::cout << "SharedMemPerBlock: " << device_prop.sharedMemPerBlock << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Max Persistent L2 Cache Size: "
              << device_prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB"
              << std::endl;

    size_t const num_megabytes_streaming_data{1024};
    if (num_megabytes_persistent_data > num_megabytes_streaming_data)
    {
        std::runtime_error(
            "Try setting persistent data size smaller than 1024 MB.");
    }
    size_t const size_persistent(num_megabytes_persistent_data * 1024 * 1024 /
                                 sizeof(int));
    size_t const size_streaming(num_megabytes_streaming_data * 1024 * 1024 /
                                sizeof(int));
    std::cout << "Persistent Data Size: " << num_megabytes_persistent_data
              << " MB" << std::endl;
    std::cout << "Steaming Data Size: " << num_megabytes_streaming_data << " MB"
              << std::endl;
    cudaStream_t stream;

    std::vector<int> lut_persistent_vec(size_persistent, 0);
    for (size_t i{0}; i < lut_persistent_vec.size(); ++i)
    {
        lut_persistent_vec[i] = i;
    }
    std::vector<int> data_streaming_vec(size_streaming, 0);

    int* d_lut_persistent;
    int* d_data_streaming;
    int* h_lut_persistent = lut_persistent_vec.data();
    int* h_data_streaming = data_streaming_vec.data();

    CHECK_CUDA_ERROR(
        cudaMalloc(&d_lut_persistent, size_persistent * sizeof(int)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_data_streaming, size_streaming * sizeof(int)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_lut_persistent, h_lut_persistent,
                                size_persistent * sizeof(int),
                                cudaMemcpyHostToDevice));

    launch_reset_data(d_data_streaming, d_lut_persistent, size_streaming,
                      size_persistent, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(h_data_streaming, d_data_streaming,
                                size_streaming * sizeof(int),
                                cudaMemcpyDeviceToHost));
    assert(verify_data(h_data_streaming, size_persistent, size_streaming));

    std::function<void(cudaStream_t)> const function{
        std::bind(launch_reset_data, d_data_streaming, d_lut_persistent,
                  size_streaming, size_persistent, std::placeholders::_1)};
    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency Without Using Persistent L2 Cache: " << latency
              << " ms" << std::endl;

    // Start to use persistent cache.
    cudaStream_t stream_persistent_cache;
    size_t const num_megabytes_persistent_cache{3};
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_persistent_cache));

    CHECK_CUDA_ERROR(
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           num_megabytes_persistent_cache * 1024 * 1024));

    cudaStreamAttrValue stream_attribute_thrashing;
    stream_attribute_thrashing.accessPolicyWindow.base_ptr =
        reinterpret_cast<void*>(d_lut_persistent);
    stream_attribute_thrashing.accessPolicyWindow.num_bytes =
        num_megabytes_persistent_data * 1024 * 1024;
    stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;
    stream_attribute_thrashing.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting;
    stream_attribute_thrashing.accessPolicyWindow.missProp =
        cudaAccessPropertyStreaming;

    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_thrashing));

    float const latency_persistent_cache_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Potentially Thrashing): "
              << latency_persistent_cache_thrashing << " ms" << std::endl;

    cudaStreamAttrValue stream_attribute_non_thrashing{
        stream_attribute_thrashing};
    stream_attribute_non_thrashing.accessPolicyWindow.hitRatio =
        std::min(static_cast<double>(num_megabytes_persistent_cache) /
                     num_megabytes_persistent_data,
                 1.0);
    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_non_thrashing));

    float const latency_persistent_cache_non_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Non-Thrashing): "
              << latency_persistent_cache_non_thrashing << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_lut_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_data_streaming));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_persistent_cache));
}
