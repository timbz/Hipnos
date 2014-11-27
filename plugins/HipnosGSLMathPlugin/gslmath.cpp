/****************************************************************************
** 
** This file is part of HiPNOS.
** 
** Copyright 2012 Helmholtz-Zentrum Dresden-Rossendorf
** 
** HiPNOS is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** HiPNOS is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with HiPNOS.  If not, see <http://www.gnu.org/licenses/>.
** 
****************************************************************************/

#include "gslmath.h"
#include "gslmatrix.h"
#include "gslvector.h"

GslMath::GslMath()
{
    currentWavetableSize = -1;
    wavetable = 0;
    workspace = 0;
}

GslMath::~GslMath()
{
    if(wavetable)
        gsl_fft_complex_wavetable_free(wavetable);
    if(workspace)
        gsl_fft_complex_workspace_free(workspace);
}

QString GslMath::getName()
{
    return "GSL Math Plugin";
}

uint GslMath::getPerformanceHint()
{
    return 10;
}

bool GslMath::isPlatformSupported()
{
    return true;
}

MathPlugin::DataType GslMath::getDataType()
{
    return DT_DOUBLE_PRECISION;
}

Matrix* GslMath::createMatrix(int row, int col)
{
   return new GslMatrix(row, col);
}

Vector* GslMath::createVector(int s)
{
    return new GslVector(s);
}

void GslMath::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
{
    for(int j = 0; j < fx->getCols(); j++)
    {
        double x = stepx * (double)(j-fx->getCols()/2);
        for(int i = 0; i < fy->getRows(); i++)
        {
            fx->set(i, j, x);
            fy->set(i, j, stepy * (double)(i-fx->getRows()/2));
        }
    }
}

void GslMath::mult(Matrix* a, Matrix* b, Matrix* c)
{
    std::complex<double> alpha(1,0);
    std::complex<double> beta(0,0);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                a->getRows(), b->getCols(), a->getCols(),
                &alpha, a->data(), a->getRows(),
                b->data(), b->getRows(),
                &beta, c->data(), c->getRows());
}

void GslMath::mult(Matrix* a, Vector* x, Vector* y)
{
    std::complex<double> alpha(1,0);
    std::complex<double> beta(0,0);
    cblas_zgemv(CblasColMajor, CblasNoTrans,
                a->getRows(), a->getCols(),
                &alpha, a->data(), a->getRows(),
                x->data(), 1, &beta, y->data(), 1);
}

void GslMath::mult(Matrix *a, std::complex<double> s)
{
    cblas_zscal(a->getRows() * a->getCols(), &s, a->data(), 1);
}

void GslMath::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    std::complex<double>* B = static_cast<std::complex<double>*>(b->data());
    std::complex<double>* C = static_cast<std::complex<double>*>(c->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        C[i] = A[i] * B[i];
    }
}

void GslMath::componentWiseMult(Matrix* a, Matrix* b)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    std::complex<double>* B = static_cast<std::complex<double>*>(b->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] *= B[i];
    }
}

void GslMath::add(Matrix *a, Matrix *b)
{
    std::complex<double> alpha(1,0);
    cblas_zaxpy(a->getRows() * a->getCols(), &alpha, b->data(), 1, a->data(), 1);
}

void GslMath::exp(Matrix *a)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] = std::exp(A[i]);
    }
}

void GslMath::pow(Matrix *a, int p)
{
    while(p > 1)
    {
        componentWiseMult(a, a);
        p--;
    }
}


void GslMath::initFft(int size)
{
    if(currentWavetableSize != size)
    {
        if(wavetable)
            gsl_fft_complex_wavetable_free(wavetable);
        if(workspace)
            gsl_fft_complex_workspace_free(workspace);
        wavetable = gsl_fft_complex_wavetable_alloc(size);
        workspace = gsl_fft_complex_workspace_alloc(size);
        currentWavetableSize = size;
    }
}

void GslMath::fft(Matrix *a)
{
    initFft(a->getRows());
    // fft over cols;
    for(int i = 0; i < a->getCols(); i++)
    {
        double* data = &static_cast<double*>(a->data())[i*2*a->getRows()];
        gsl_fft_complex_forward(data, 1, a->getRows(), wavetable, workspace);
    }
    // fft over rows;
    initFft(a->getCols());
    for(int i = 0; i < a->getRows(); i++)
    {
        double* data = &static_cast<double*>(a->data())[i*2];
        gsl_fft_complex_forward(data, a->getRows(), a->getCols(), wavetable, workspace);
    }
}

void GslMath::ifft(Matrix *a)
{
    //NOTE: gsl_fft_complex_inverse scales the result by  1/n, gsl_fft_complex_backward does no scaling

    initFft(a->getRows());
    // fft over cols;
    for(int i = 0; i < a->getCols(); i++)
    {
        double* data = &static_cast<double*>(a->data())[i*2*a->getRows()];
        gsl_fft_complex_inverse(data, 1, a->getRows(), wavetable, workspace);
    }
    // fft over rows;
    initFft(a->getCols());
    for(int i = 0; i < a->getRows(); i++)
    {
        double* data = &static_cast<double*>(a->data())[i*2];
        gsl_fft_complex_inverse(data, a->getRows(), a->getCols(), wavetable, workspace);
    }
}

void GslMath::fft(Matrix *in, Matrix *out)
{
}

void GslMath::ifft(Matrix *in, Matrix *out)
{

}

void GslMath::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols()% 2 == 0);

    int N = a->getRows()/2;
    int M = a->getCols()/2;
    for(int col = 0; col < M; col++)
    {
        void* X = &static_cast<std::complex<double>*>(a->data())[col * a->getRows()];
        void* Y = &static_cast<std::complex<double>*>(a->data())[(M + col) * a->getRows() + N];
        cblas_zswap(N, X, 1, Y, 1);

        X = &static_cast<std::complex<double>*>(a->data())[col * a->getRows() + N];
        Y = &static_cast<std::complex<double>*>(a->data())[(M + col) * a->getRows()];
        cblas_zswap(N, X, 1, Y, 1);
    }
}

void GslMath::ifftshift(Matrix *a)
{
    fftshift(a);
}

 Q_EXPORT_PLUGIN2(hipnosgslmathplugin, GslMath);
