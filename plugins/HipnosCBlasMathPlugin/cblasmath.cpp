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

#include <QProgressDialog>

#include <cblas.h>
#include <fftw3.h>

#include "cblasmath.h"
#include "cblasmatrix.h"
#include "cblasvector.h"
#include "common/hipnossettings.h"

CBlasMath::CBlasMath()
{
    wisdomFileName = QDir::home().path() + QDir::separator() + HIPNOS_HOME_DIR_NAME + QDir::separator() + "fftw.wisdom";

#ifdef QT_NO_DEBUG
    loadWisdom();
    generatePlans();
    saveWisdom();
#endif
}

CBlasMath::~CBlasMath()
{
}

void CBlasMath::loadWisdom()
{
    // Older versions of fftw3 dont have the this method
//    int err = fftw_import_wisdom_from_filename(wisdomFileName.toLocal8Bit().constData());
//    if(err != 1)
//    {
//           qWarning(QString("fftw_import_wisdom_from_filename returned " + QString::number(err) +
//                            " while trying to read wisdoms from " + wisdomFileName).toLocal8Bit().constData());
//    }
    QFile file(wisdomFileName);
    if(file.exists())
    {
        qDebug() << "Loading wisdoms ...";
        if(!file.open(QIODevice::ReadOnly))
        {
            qWarning() << ("Error opening file " + wisdomFileName);
            return;
        }
        FILE *f = fdopen(file.handle(), "r");
        if (!f)
        {
            qDebug() << ("Error opening file " + wisdomFileName);
            return;
        }
        int err = fftw_import_wisdom_from_file(f);
        if(err != 1)
        {
            qWarning(QString("fftw_import_wisdom_from_filename returned " + QString::number(err) +
                              " while trying to read wisdoms from " + wisdomFileName).toLocal8Bit().constData());
        }
        if (fclose(f))
        {
            qWarning("Error closing file");
        }
    }
    else
    {
        qDebug() << "No wisdom file found";
    }
}

void CBlasMath::generatePlans()
{
    unsigned int plannerMode = FFTW_EXHAUSTIVE;
    int minN = 2;
    int maxN = 1024;

    if(qApp) // test we have are in a gui env
    {
        QProgressDialog progress("CBlas Math Plugin is tuning fftw ...", "Abort", minN, maxN);
        progress.setMinimumDuration(0);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();

        for(int n = minN; n <= maxN; n++)
        {
            progress.setValue(n);
            progress.setLabelText("Tuning fftw for matrix size " + QString::number(n) + " ...");
            fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);
            fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);

            qApp->processEvents();
            fftw_plan pf = fftw_plan_dft_2d(n, n, in, out, FFTW_FORWARD, plannerMode);
            if (progress.wasCanceled())
                         break;

            qApp->processEvents();
            fftw_plan pb = fftw_plan_dft_2d(n, n, in, out, FFTW_BACKWARD, plannerMode);
            if (progress.wasCanceled())
                         break;

            fftw_destroy_plan(pf);
            fftw_destroy_plan(pb);
            fftw_free(in);
            fftw_free(out);
        }
    }
    else
    {
//        std::cout << "CBlas Math Plugin is tuning fftw ... (press ENTER to cancel)"  << std::endl;
//        for(int n = minN; n <= maxN; n++)
//        {
//            std::cout << '\xd' << "Tuning fftw for matrix size " << n;
//            //std::cout << std::endl << "Tuning fftw for matrix size" << n;
//            fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);
//            fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n);

//            if(cinThread.isFinished())
//                break;
//            fftw_plan pf = fftw_plan_dft_2d(n, n, in, out, FFTW_FORWARD, plannerMode);

//            if(cinThread.isFinished())
//                break;
//            fftw_plan pb = fftw_plan_dft_2d(n, n, in, out, FFTW_BACKWARD, plannerMode);

//            fftw_destroy_plan(pf);
//            fftw_destroy_plan(pb);
//            fftw_free(in);
//            fftw_free(out);
//        }
//        std::cout << std::endl << "Done!" << std::endl;
    }
}

void CBlasMath::saveWisdom()
{

    // Older versions of fftw3 dont have the this method
//    int err = fftw_export_wisdom_to_filename(wisdomFileName.toLocal8Bit().constData());
//    if(err != 1)
//    {
//        qCritical(QString("fftw_export_wisdom_to_filename returned " + QString::number(err) +
//                         " while trying to write wisdoms to " + wisdomFileName).toLocal8Bit().constData());
//    }
    QFile file(wisdomFileName);
    qDebug() << "Saving wisdoms ...";
    if(!file.open(QIODevice::WriteOnly))
    {
        qWarning() << ("Error opening file " + wisdomFileName);
        return;
    }
    FILE *f = fdopen(file.handle(), "w");
    if (!f)
    {
        qWarning() << ("Error opening file " + wisdomFileName);
        return;
    }
    fftw_export_wisdom_to_file(f);
    if(ferror(f) != 0)
    {
        qWarning("Error writing to file");
    }
    if (fclose(f))
    {
        qWarning("Error closing file");
    }
}

QString CBlasMath::getName()
{
    return "CBlas Math Plugin";
}

uint CBlasMath::getPerformanceHint()
{
    return 50;
}

bool CBlasMath::isPlatformSupported()
{
    return true;
}

MathPlugin::DataType CBlasMath::getDataType()
{
    return DT_DOUBLE_PRECISION;
}

Matrix* CBlasMath::createMatrix(int row, int col)
{
   return new CBlasMatrix(row, col);
}

Vector* CBlasMath::createVector(int s)
{
    return new CBlasVector(s);
}

void CBlasMath::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
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

void CBlasMath::mult(Matrix* a, Matrix* b, Matrix* c)
{
    std::complex<double> alpha(1,0);
    std::complex<double> beta(0,0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                a->getRows(), b->getCols(), a->getCols(),
                &alpha, a->data(), a->getRows(),
                b->data(), b->getRows(),
                &beta, c->data(), c->getRows());
}

void CBlasMath::mult(Matrix* a, Vector* x, Vector* y)
{
    std::complex<double> alpha(1,0);
    std::complex<double> beta(0,0);
    cblas_zgemv(CblasColMajor, CblasNoTrans,
                a->getRows(), a->getCols(),
                &alpha, a->data(), a->getRows(),
                x->data(), 1, &beta, y->data(), 1);
}

void CBlasMath::mult(Matrix *a, std::complex<double> s)
{
    cblas_zscal(a->getRows() * a->getCols(), &s, a->data(), 1);
}

void CBlasMath::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
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

void CBlasMath::componentWiseMult(Matrix* a, Matrix* b)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    std::complex<double>* B = static_cast<std::complex<double>*>(b->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] *= B[i];
    }
}

void CBlasMath::add(Matrix *a, Matrix *b)
{
    std::complex<double> alpha(1,0);
    cblas_zaxpy(a->getRows() * a->getCols(), &alpha, b->data(), 1, a->data(), 1);
}

void CBlasMath::exp(Matrix *a)
{
    std::complex<double>* A = static_cast<std::complex<double>*>(a->data());
    int size = a->getRows() * a->getCols();
    for(int i = 0; i < size; i++)
    {
        A[i] = std::exp(A[i]);
    }
}

void CBlasMath::pow(Matrix *a, int p)
{
    while(p > 1)
    {
        componentWiseMult(a, a);
        p--;
    }
}

void CBlasMath::fft(Matrix *a, int direction)
{
    lock.lock();
    fftw_plan p = fftw_plan_dft_2d(a->getCols(), a->getRows(), reinterpret_cast<fftw_complex*>(a->data()),
                                   reinterpret_cast<fftw_complex*>(a->data()),
                                   direction, FFTW_ESTIMATE);
    lock.unlock();

    fftw_execute(p);

    lock.lock();
    fftw_destroy_plan(p);
    lock.unlock();
}

void CBlasMath::fft(Matrix *a)
{
    fft(a, FFTW_FORWARD);
}

void CBlasMath::ifft(Matrix *a)
{
    fft(a, FFTW_BACKWARD);
    std::complex<double> N(1.0/(a->getRows()*a->getCols()));
    cblas_zscal(a->getRows()*a->getCols(), &N, a->data(), 1);
}

void CBlasMath::fft(Matrix *in, Matrix *out)
{
}

void CBlasMath::ifft(Matrix *in, Matrix *out)
{
}

void CBlasMath::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols()% 2 == 0);

    int N = a->getRows()/2;
    int M = a->getCols()/2;
    for(int col = 0; col <M; col++)
    {
        void* X = &static_cast<std::complex<double>*>(a->data())[col * a->getRows()];
        void* Y = &static_cast<std::complex<double>*>(a->data())[(M + col) * a->getRows() + N];
        cblas_zswap(N, X, 1, Y, 1);

        X = &static_cast<std::complex<double>*>(a->data())[col * a->getRows() + N];
        Y = &static_cast<std::complex<double>*>(a->data())[(M + col) * a->getRows()];
        cblas_zswap(N, X, 1, Y, 1);
    }
}

void CBlasMath::ifftshift(Matrix *a)
{
    fftshift(a);
}

Q_EXPORT_PLUGIN2(hipnoscblasmathplugin, CBlasMath);
