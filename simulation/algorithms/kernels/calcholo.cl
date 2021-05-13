/*
--> The following code for calculating holographic images
    is based on the ideas and development of GRIER G.
	https://github.com/davidgrier/lorenzmie/

--> Kernel file received from THALHAMMER G.

CHANGES / ADDITIONAL FUNCTIONALITY by ZOBERNIG M.

 + redefinition of the reference wave
     -> phase=0 at z=0
     -> Focus point of the camera as reference point

 + generalization of the image input shape to any possible given shape
     -> square images and rectangular images of all shapes can now be calculated

 + definition of 0/0 singularities appearing in trigonometric definitions
     -> introduction of limits that fix this problem

 + generalization of the input parameters
     -> any given number of particles in an image are now considered and
         calculated properly
     -> routine is running on local variable X, Y, Z, Inc to reduce number
         of global memory accesses

 + accelleration of the kernel by initially calculating frequently occuring
   variables that are needed more than one time

 + generalization of the linear polarized incoming wave

 + generalization of the wave vector (vector in spherical coordinates)

 + giving flexibility by calculating individual fields or intensity itself
    -> reduces number of global memory accesses if wanted

 + introduction of a simple model to take incoherence of the incoming
   light wave into account

 + separation into different functions
    -> main routine "cl_calc"
    -> sub routine "cl_holo" that calculates whole image of given shape
    -> sub routine "cl_holo_percentage" that calculates only parts of 
        a given image (dependent on the percentage of the input shape)

 + introduction of the helper function "abs2"
*/

typedef float2 cmplx;
#define PI_F 3.1415926535897932384626f

// complex multiplication
inline cmplx cmul(cmplx a, cmplx b) {
    return (cmplx)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// calculation of the vector norm
inline float abs2(cmplx a, cmplx b, cmplx c) {
    return (a.x*a.x+a.y*a.y + b.x*b.x+b.y*b.y + c.x*c.x+c.y*c.y);
}

// calculation of (A*B).real
inline float Re_cmul_t(cmplx a, cmplx b) {
    return (a.x*b.x + a.y*b.y);
}

// main calculation routine for N particles
inline void cl_calc(float x,
                    float y,
                    unsigned int gid,
                    const int Nmax,
                    const int number,
                    const int offset,
                    const float lamda,
                    const float n1,
                    const float lc,
                    const int coherence,
                    const int full,
                  __global const float4* angles,
                  __global const float4* parameters,
                  __global const cmplx* a,
                  __global const cmplx* b,
                  __global cmplx* EX,
                  __global cmplx* EY,
                  __global cmplx* EZ,
                  __global float* I)
{
    // wavenumber
    float k = (2*PI_F)/(lamda/n1);
    // sigma of a Gaussian function, connection with the coherence length
    float sigma = lc / 2.3548;
    
    // incident wave angles for polarization and orientation
    float4 angles_received = angles[0];
    float Phi0 = angles_received.x;
    float Theta0 = angles_received.y;
    float Psi0 = angles_received.z;
    
    float cos_Phi0 = cos(Phi0);
    float sin_Phi0 = sin(Phi0);
    float cos_Theta0 = cos(Theta0);
    float sin_Theta0 = sin(Theta0);
    float cos_Psi0 = cos(Psi0);
    float sin_Psi0 = sin(Psi0);
    
    // local field variables
    float Inc = 0;
    
    // linearly polarized wave with angle Psi with respect to x axis
    cmplx X = (cmplx)(cos_Psi0, 0);
    cmplx Y = (cmplx)(sin_Psi0, 0);
    cmplx Z = (cmplx)(0,        0);

    // iteration of number of particles, saved into local variables
    for (int m=0; m<number; m++) {
        // particle parameters
        float4 pos = parameters[m+offset];
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float alpha = pos.w;

        // relative position
        float xn = x-px;
        float yn = y-py;

        float rho = sqrt(xn*xn + yn*yn);
        float r = sqrt(xn*xn + yn*yn + pz*pz);
        float delta_r = r-pz;

        // definition of the trigonometric function values
        float cos_phi = (rho==0) ? 1 : xn/rho;
        float sin_phi = (rho==0) ? 0 : yn/rho;
        float cos_theta = pz/r;
        float sin_theta = rho/r;

        // creation of the reference wave in the particle position
        //with arbitrary wave vector n
        float nR =  sin_Theta0*cos_Phi0*px +
                    sin_Theta0*sin_Phi0*py +
                    cos_Theta0*pz;
        float KR = nR*k;
        
        cmplx Ep = (cmplx)(cos(KR), -sin(KR));

        //starting points for recursion
        float kr = k*r;
        cmplx xi_n_m2 = (cmplx)(cos(kr), sin(kr));
        cmplx xi_n_m1 = (cmplx)(sin(kr),-cos(kr));
        cmplx xi_n;

        float pi_n_m1 = 0.f;
        float pi_n = 1.f;

        cmplx Es_r=0, Es_th=0, Es_ph=0;
        cmplx Mo1n_r, Mo1n_th, Mo1n_ph;
        cmplx Ne1n_r, Ne1n_th, Ne1n_ph;

        float swisc, twisc, tau_n;
        cmplx dn;
        cmplx En;
        cmplx i_hoch_n = (cmplx)(0, 1.f);

        // recursion
        int index;
        for (int n=1; n<=Nmax; n++) {
            swisc = pi_n * cos_theta;
            twisc = swisc - pi_n_m1;
            tau_n = pi_n_m1 - n*twisc;

            xi_n = (2*n-1)*xi_n_m1/kr - xi_n_m2;

            Mo1n_r = 0.f;
            Mo1n_th = pi_n * xi_n;
            Mo1n_ph = tau_n * xi_n;

            dn = (n*xi_n)/kr - xi_n_m1;
            Ne1n_r = n*(n+1)*pi_n*xi_n;
            Ne1n_th = tau_n*dn;
            Ne1n_ph = pi_n*dn;

            En = i_hoch_n * (2*n+1.f)/((float)n*((float)n+1.f));

            index = n+(m+offset)*(Nmax+1);
            Es_r  += cmul(cmul(cmul(En, (cmplx)(0,1.f)),a[index]), Ne1n_r)  -
                    cmul(cmul(En, b[index]), Mo1n_r);
            Es_th += cmul(cmul(cmul(En, (cmplx)(0,1.f)),a[index]), Ne1n_th) -
                    cmul(cmul(En, b[index]), Mo1n_th);
            Es_ph += cmul(cmul(cmul(En, (cmplx)(0,1.f)),a[index]), Ne1n_ph) -
                    cmul(cmul(En, b[index]), Mo1n_ph);

            pi_n_m1 = pi_n;
            pi_n = swisc + ((n+1.f)/(float)n)*twisc;

            xi_n_m2 = xi_n_m1;
            xi_n_m1 = xi_n;

            i_hoch_n = cmul(i_hoch_n, (cmplx)(0,1.f));
        }

        // scattered fields
        Es_r  *= cos_phi*sin_theta / (kr*kr);
        Es_th *= cos_phi/kr;
        Es_ph *= sin_phi/kr;

        // recalculation in the X, Y, Z coordinate system
        cmplx Ec_x =   Es_r*sin_theta*cos_phi
                     + Es_th*cos_theta*cos_phi
                     - Es_ph*sin_phi;

        cmplx Ec_y =   Es_r*sin_theta*sin_phi
                     + Es_th*cos_theta*sin_phi
                     + Es_ph*cos_phi;

        cmplx Ec_z =   Es_r*cos_theta
                     - Es_th*sin_theta;
        
        // polarization correction
        Ec_x *= cos_Psi0;
        Ec_y *= sin_Psi0;
        
        // create interference
        Ec_x = cmul(Ep, Ec_x);
        Ec_y = cmul(Ep, Ec_y);
        Ec_z = cmul(Ep, Ec_z);
        
        // correction with the alpha value (global contrast parameter)
        Ec_x *= alpha;
        Ec_y *= alpha;
        Ec_z *= alpha;

        // consideration of coherence length of the light
        if (coherence == 1) {
            float f0 = exp((delta_r*delta_r)/(-2*sigma*sigma));
            float f1 = sqrt(1-f0*f0);

            Ec_x *= f0;
            Ec_y *= f0;
            Ec_z *= f0;
            Inc += f1*abs2(Ec_x, Ec_y, Ec_z);
        }

        // add fields to local variables
        X += Ec_x;
        Y += Ec_y;
        Z += Ec_z;
    } // end of particle iteration
    
    // if full field information is needed
    if (full == 1) {
        EX[gid] = X;
        EY[gid] = Y;
        EZ[gid] = Z;
        I[gid] = Inc;
    } else {
        I[gid] = Re_cmul_t(X, X) +
                 Re_cmul_t(Y, Y) +
                 Re_cmul_t(Z, Z) +
                 Inc;
    }
}

// complete function that calculates every pixel of an image
__kernel void cl_holo(const int Nmax,
                      const int number,
                      const int offset,
                      const float dx,
                      const float lamda,
                      const float n1,
                      const float lc,
                      const int coherence,
                      const int full,
                     __global const float4* angles,
                     __global const float4* parameters,
                     __global const cmplx* a,
                     __global const cmplx* b,
                     __global cmplx* EX,
                     __global cmplx* EY,
                     __global cmplx* EZ,
                     __global float* I)
{
    unsigned int xi = get_global_id(1);
    unsigned int yi = get_global_id(0);
    unsigned int gid = xi + yi*get_global_size(1);

    float x = dx*(xi-get_global_size(1)*0.5f);
    float y = dx*(yi-get_global_size(0)*0.5f);

    cl_calc(x, y, gid, Nmax, number, offset, lamda, n1, lc, coherence, full, angles, 
    parameters, a, b, EX, EY, EZ, I);
}

// partial calculation of an image with a percentage of all pixels
__kernel void cl_holo_percentage(const int Nmax,
                      const int number,
                      const int offset,
                      const float dx,
                      const float lamda,
                      const float n1,
                      const float lc,
                      const int coherence,
                      const int full,
                     __global const float4* angles,
                     __global const float4* parameters,
                     __global const cmplx* a,
                     __global const cmplx* b,
                     __global cmplx* EX,
                     __global cmplx* EY,
                     __global cmplx* EZ,
                     __global float* I,
                     __global const int* selr,
                     __global const int* selc,
                     const float Nrow,
                     const float Ncol)
{
    unsigned int gid = get_global_id(0);
    unsigned int xi = selc[gid];
    unsigned int yi = selr[gid];

    float x = dx*(xi-Ncol*0.5f);
    float y = dx*(yi-Nrow*0.5f);

    cl_calc(x, y, gid, Nmax, number, offset, lamda, n1, lc, coherence, full, angles, 
    parameters, a, b, EX, EY, EZ, I);
}