#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <iostream>

struct complex_exp_functor : public thrust::unary_function<cuComplex,cuComplex>
{
    __host__ __device__ cuComplex operator()(const cuComplex &arg) const
    {
        //exp(z) = exp(x) * (cos(y) + i * sin(y))
        float e = exp(arg.x);
        float s, c;
        sincos(arg.y, &s, &c);
        return make_cuComplex(c * e, s * e);
    }
};

struct comlex_mult_functor : public thrust::binary_function<cuComplex,cuComplex,cuComplex>
{
    __host__ __device__ cuComplex operator()(const cuComplex &lhs, const cuComplex &rhs) const
    {
        return make_cuComplex((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
    }
};

struct complex_sequence_functor
{
    const float init;
    const float step;

    complex_sequence_functor(float _init, float _step)
        : init(_init), step(_step) {}

    __host__ __device__ cuComplex operator()(const int i) const
    {
        return make_cuComplex(init + step * i, 0);
    }
};

extern "C"
cudaError_t cuda_exp(cuComplex* data, int size)
{
    thrust::device_ptr<cuComplex> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, complex_exp_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_mult_inplace(cuComplex* data1, cuComplex* data2, int size)
{
    thrust::device_ptr<cuComplex> dev_ptr1(data1);
    thrust::device_ptr<cuComplex> dev_ptr2(data2);
    thrust::transform(dev_ptr1, dev_ptr1 + size, dev_ptr2, dev_ptr1, comlex_mult_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_mult(cuComplex* data1, cuComplex* data2, cuComplex* result, int size)
{
    thrust::device_ptr<cuComplex> dev_ptr1(data1);
    thrust::device_ptr<cuComplex> dev_ptr2(data2);
    thrust::device_ptr<cuComplex> dev_ptr3(result);
    thrust::transform(dev_ptr1, dev_ptr1 + size, dev_ptr2, dev_ptr3, comlex_mult_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_pow(cuComplex* data1, int size, int pow)
{
    thrust::device_ptr<cuComplex> dev_ptr(data1);
    while(pow > 1)
    {
        thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, dev_ptr, comlex_mult_functor());
        pow--;
    }
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_for_matrix(cuComplex* fx, cuComplex* fy, int rows, int cols, float stepx, float stepy)
{
    thrust::device_ptr<cuComplex> dev_ptr_fx(fx);
    thrust::device_ptr<cuComplex> dev_ptr_fy(fy);
    for(int c = 0; c < cols; c++)
    {
        thrust::fill(dev_ptr_fx + c*rows, dev_ptr_fx + c*rows + rows, make_cuComplex(stepx * (float)(c-cols/2), 0.0));
        thrust::counting_iterator<int> iter = thrust::make_counting_iterator(0);
        thrust::transform(iter, iter + rows, dev_ptr_fy + c*rows, complex_sequence_functor(stepy * (float)(-rows/2), stepy));
    }
    return cudaGetLastError();
}
