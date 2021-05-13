/*
The following code is used to let the arrays lie on the graphics card
and modify it right there.
*/

typedef float2 cmplx;

// calculation of (A*B).real
inline float Re_cmul_t(cmplx a, cmplx b) {
    return (a.x*b.x + a.y*b.y);
}

// complex multiplication
inline cmplx cmul(cmplx a, cmplx b) {
    return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// complex multiplication kernel with mask array2
__kernel void i_mult(__global cmplx* array1,
                     __global int* array2)
{
    unsigned int xi = get_global_id(1);
    unsigned int yi = get_global_id(0);
    unsigned int gid = xi + yi*get_global_size(1);
    array1[gid] *= array2[gid];
}

// complex multiplication kernel with phase shift array
__kernel void c_mult(__global cmplx* array1,
                     __global cmplx* array2)
{
    unsigned int xi = get_global_id(1);
    unsigned int yi = get_global_id(0);
    unsigned int gid = xi + yi*get_global_size(1);
    array1[gid] = cmul(array1[gid], array2[gid]);
}

// real multiplication kernel field array into intensity array
__kernel void re_mult(__global float* array1,
                      __global cmplx* array2)
{
    unsigned int xi = get_global_id(1);
    unsigned int yi = get_global_id(0);
    unsigned int gid = xi + yi*get_global_size(1);
    array1[gid] += Re_cmul_t(array2[gid], array2[gid]);
}