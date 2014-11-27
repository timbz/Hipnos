#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void complex_add(__global double2* a, __global double2* b)
{
    unsigned int n = get_global_id(0);
    a[n].x += b[n].x;
    a[n].y += b[n].y;
}

__kernel void complex_mult(__global double2* a, __global double2* b, __global double2* c)
{
    unsigned int n = get_global_id(0);
    c[n].x = a[n].x * b[n].x - a[n].y * b[n].y;
    c[n].y = a[n].x * b[n].y + a[n].y * b[n].x;
}

__kernel void complex_scalar_mult(__global double2* a, double2 b)
{
    unsigned int n = get_global_id(0);
    a[n].x = a[n].x * b.x - a[n].y * b.y;
    a[n].y = a[n].x * b.y + a[n].y * b.x;
}

__kernel void complex_exp(__global double2* a)
{
    unsigned int n = get_global_id(0);
    double e = exp(a[n].x);
    double s, c;
    s = sincos(a[n].y, &c);
    a[n].x = c * e;
    a[n].y = s * e;
}

__kernel void for_matrices(__global double2* fx, __global double2* fy, int rows, int cols, double stepx, double stepy)
{
    unsigned int n = get_global_id(0);
    int j = n / rows;
    int i = n % rows;
    fx[n].x = stepx * (double)(j-cols/2);
    fx[n].y = 0.0;
    fy[n].x = stepy * (double)(i-rows/2);
    fy[n].y = 0.0;
}

__kernel void complex_fftshift(__global double2* a, int rows, int cols)
{
    unsigned int n = get_global_id(0);
    if(n < rows * cols / 2)
    {
        int i = n % rows;
        int m;
        if(i < rows/2)
        {
            m = n + rows*cols/2 + rows/2;
        }
        else
        {
            m = n + rows*cols/2 - rows/2;
        }
        double2 tmp = a[n];
        a[n] = a[m];
        a[m] = tmp;
    }
}
