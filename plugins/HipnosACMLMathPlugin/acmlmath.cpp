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

#include "acmlmath.h"
#include "acmlmatrix.h"
#include "acmlvector.h"

AcmlMath::AcmlMath()
{
    //TODO: optimize execution by saving and loading comm array from file
    currentCommArrayMatrixSize = QSize(-1,-1);
    comm = 0;
}

AcmlMath::~AcmlMath()
{
    if(comm)
        free(comm);
}

QString AcmlMath::getName()
{
    return "AMD ACML Math Plugin";
}

uint AcmlMath::getPerformanceHint()
{
    return 55;
}

bool AcmlMath::isPlatformSupported()
{
    return true;
}

MathPlugin::DataType AcmlMath::getDataType()
{
    return DT_DOUBLE_PRECISION;
}

Matrix* AcmlMath::createMatrix(int row, int col)
{
   return new AcmlMatrix(row, col);
}

Vector* AcmlMath::createVector(int s)
{
    return new AcmlVector(s);
}

void AcmlMath::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
{
    for(int j = 0; j < fx->getCols(); j++)
    {
        double x = stepx * (double)(j-fx->getCols()/2);
        for(int i = 0; i < fx->getRows(); i++)
        {
            fx->set(i, j, x);
            fy->set(i, j, stepy * (double)(i-fx->getRows()/2));
        }
    }
}

void AcmlMath::mult(Matrix* a, Matrix* b, Matrix* c)
{
    doublecomplex alpha = compose_doublecomplex(1,0);
    doublecomplex beta = compose_doublecomplex(0,0);
    doublecomplex* A = static_cast<doublecomplex*>(a->data());
    doublecomplex* B = static_cast<doublecomplex*>(b->data());
    doublecomplex* C = static_cast<doublecomplex*>(c->data());

    zgemm('N', 'N',
            a->getRows(), b->getCols(), a->getCols(),
            &alpha, A, a->getRows(),
            B, b->getRows(),
            &beta, C, c->getRows());
}

void AcmlMath::mult(Matrix* a, Vector* x, Vector* y)
{
    doublecomplex alpha = compose_doublecomplex(1,0);
    doublecomplex beta = compose_doublecomplex(0,0);
    doublecomplex* A = static_cast<doublecomplex*>(a->data());
    doublecomplex* X = static_cast<doublecomplex*>(x->data());
    doublecomplex* Y = static_cast<doublecomplex*>(y->data());
    zgemv('N',
            a->getRows(), a->getCols(),
            &alpha, A, a->getRows(),
            X, 1, &beta, Y, 1);
}

void AcmlMath::mult(Matrix *a, std::complex<double> s)
{
    doublecomplex* A = static_cast<doublecomplex*>(a->data());
    doublecomplex alpha = compose_doublecomplex(s.real(), s.imag());
    zscal(a->getRows() * a->getCols(), &alpha, A, 1);
}

void AcmlMath::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
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

void AcmlMath::componentWiseMult(Matrix* a, Matrix* b)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    std::complex<double>* B = static_cast<std::complex<double>*>(b->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] *= B[i];
    }
}

void AcmlMath::add(Matrix *a, Matrix *b)
{
    doublecomplex alpha = compose_doublecomplex(1,0);
    doublecomplex* A = static_cast<doublecomplex*>(a->data());
    doublecomplex* B = static_cast<doublecomplex*>(b->data());
    zaxpy(a->getRows() * a->getCols(), &alpha, B, 1, A, 1);
}

void AcmlMath::exp(Matrix *a)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] = std::exp(A[i]);
    }
}

void AcmlMath::pow(Matrix *a, int p)
{
    while(p > 1)
    {
        componentWiseMult(a, a);
        p--;
    }
}

void AcmlMath::fft(Matrix *a, int direction)
{
    if(currentCommArrayMatrixSize != QSize(a->getRows(), a->getRows()))
    {
        if(comm)
            free(comm);
        comm = (doublecomplex *)malloc((a->getRows()*a->getCols()+3*(a->getRows()+a->getCols())+100)*sizeof(doublecomplex));
        currentCommArrayMatrixSize = QSize(a->getRows(), a->getRows());
    }
    int info;
    double scale = 1;
    zfft2d(direction, a->getCols(), a->getRows(), static_cast<doublecomplex*>(a->data()), comm, &info);
    //if(direction > 0)
    //    scale = a->getCols()*a->getCols();
    //zfft2dx(direction*2, scale, 1, 1, a->getRows(), a->getCols(), static_cast<doublecomplex*>(a->data()), 1, 1, static_cast<doublecomplex*>(a->data()), 1, 1, comm, &info);
    if(info != 0)
    {
        QString error = "ACMLs zfft2d failed (code " + QString::number(info) + ")";
        qFatal(error.toStdString().c_str());
    }
}

void AcmlMath::fft(Matrix *a)
{
    fft(a, -1);
}

void AcmlMath::ifft(Matrix *a)
{
    fft(a, 1);
}

void AcmlMath::fft(Matrix *in, Matrix *out)
{
}

void AcmlMath::ifft(Matrix *in, Matrix *out)
{
}

void AcmlMath::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols()% 2 == 0);

    int N = a->getRows()/2;
    int M = a->getCols()/2;
    for(int col = 0; col <M; col++)
    {
        doublecomplex* X = &static_cast<doublecomplex*>(a->data())[col * a->getRows()];
        doublecomplex* Y = &static_cast<doublecomplex*>(a->data())[(M + col) * a->getRows() + N];
        zswap(N, X, 1, Y, 1);

        X = &static_cast<doublecomplex*>(a->data())[col * a->getRows() + N];
        Y = &static_cast<doublecomplex*>(a->data())[(M + col) * a->getRows()];
        zswap(N, X, 1, Y, 1);
    }
}

void AcmlMath::ifftshift(Matrix *a)
{
    fftshift(a);
}

Q_EXPORT_PLUGIN2(hipnosacmlmathplugin, AcmlMath);
