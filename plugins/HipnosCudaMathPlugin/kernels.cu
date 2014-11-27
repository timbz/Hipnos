#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <iostream>

struct complex_exp_functor : public thrust::unary_function<cuDoubleComplex,cuDoubleComplex>
{
    __host__ __device__ cuDoubleComplex operator()(const cuDoubleComplex &arg) const
    {
        //exp(z) = exp(x) * (cos(y) + i * sin(y))
        double e = exp(arg.x);
        double s, c;
        sincos(arg.y, &s, &c);
        return make_cuDoubleComplex(c * e, s * e);
    }
};

struct comlex_mult_functor : public thrust::binary_function<cuDoubleComplex,cuDoubleComplex,cuDoubleComplex>
{
    __host__ __device__ cuDoubleComplex operator()(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) const
    {
        return make_cuDoubleComplex((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
    }
};

struct complex_sequence_functor
{
    const double init;
    const double step;

    complex_sequence_functor(double _init, double _step)
        : init(_init), step(_step) {}

    __host__ __device__ cuDoubleComplex operator()(const int i) const
    {
        return make_cuDoubleComplex(init + step * i, 0);
    }
};

extern "C"
cudaError_t cuda_exp(cuDoubleComplex* data, int size)
{
    thrust::device_ptr<cuDoubleComplex> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, complex_exp_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_mult_inplace(cuDoubleComplex* data1, cuDoubleComplex* data2, int size)
{
    thrust::device_ptr<cuDoubleComplex> dev_ptr1(data1);
    thrust::device_ptr<cuDoubleComplex> dev_ptr2(data2);
    thrust::transform(dev_ptr1, dev_ptr1 + size, dev_ptr2, dev_ptr1, comlex_mult_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_mult(cuDoubleComplex* data1, cuDoubleComplex* data2, cuDoubleComplex* result, int size)
{
    thrust::device_ptr<cuDoubleComplex> dev_ptr1(data1);
    thrust::device_ptr<cuDoubleComplex> dev_ptr2(data2);
    thrust::device_ptr<cuDoubleComplex> dev_ptr3(result);
    thrust::transform(dev_ptr1, dev_ptr1 + size, dev_ptr2, dev_ptr3, comlex_mult_functor());
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_pow(cuDoubleComplex* data1, int size, int pow)
{
    thrust::device_ptr<cuDoubleComplex> dev_ptr(data1);
    while(pow > 1)
    {
        thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, dev_ptr, comlex_mult_functor());
        pow--;
    }
    return cudaGetLastError();
}

extern "C"
cudaError_t cuda_for_matrix(cuDoubleComplex* fx, cuDoubleComplex* fy, int rows, int cols, double stepx, double stepy)
{
    thrust::device_ptr<cuDoubleComplex> dev_ptr_fx(fx);
    thrust::device_ptr<cuDoubleComplex> dev_ptr_fy(fy);
    for(int c = 0; c < cols; c++)
    {
        thrust::fill(dev_ptr_fx + c*rows, dev_ptr_fx + c*rows + rows, make_cuDoubleComplex(stepx * (double)(c-cols/2), 0.0));
        thrust::counting_iterator<int> iter = thrust::make_counting_iterator(0);
        thrust::transform(iter, iter + rows, dev_ptr_fy + c*rows, complex_sequence_functor(stepy * (double)(-rows/2), stepy));
    }
    return cudaGetLastError();
}
