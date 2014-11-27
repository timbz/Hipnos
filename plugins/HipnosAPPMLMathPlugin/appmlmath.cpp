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

#include "appmlmath.h"
#include "appmlmatrix.h"
#include "appmlvector.h"
#include "openclplatformanddevicedialog.h"

#include <QHash>
#include <QPair>

AppmlMath::AppmlMath()
{
    event = NULL;
    queue = NULL;
    ctx = NULL;
    currentFftPlanSize = QSize(-1,-1);
    plan = 0;

    /* platforms */
    const cl_uint maxNumPlatforms = 8;
    cl_platform_id platformIds[maxNumPlatforms];
    cl_uint numPlatforms;

    /* devices */
    const cl_uint maxNumDevices = 8;
    cl_device_id deviceIds[maxNumDevices];
    cl_uint numDevices;

    /* info */
    cl_device_type infoType;
    char infoStr[1024];
    size_t infoStrLen;

    /* get available platforms */
    err = clGetPlatformIDs(maxNumPlatforms, platformIds, &numPlatforms);
    checkForError("clGetPlatformIDs");

    bool atLeastOneDeviceFound = false;
    /* loop over available platforms */
    for(unsigned int platformId = 0; platformId < numPlatforms; platformId++) {

        // we could also query CL_PLATFORM_VENDOR, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS, CL_PLATFORM_VERSION
        err = clGetPlatformInfo ( platformIds[platformId], CL_PLATFORM_NAME, sizeof(infoStr), infoStr, &infoStrLen );
        checkForError("clGetPlatformInfo[CL_PLATFORM_NAME,]");
        QString platformName = QString::fromLocal8Bit(infoStr, infoStrLen-1);

        /* get the list of available devices per platform */
        err = clGetDeviceIDs( platformIds[platformId], CL_DEVICE_TYPE_ALL, maxNumDevices, deviceIds, &numDevices);
        checkForError("clGetDeviceIDs(" + platformName + ")");

        /* loop over available devices */
        for(unsigned int deviceId = 0; deviceId < numDevices; deviceId++) {

            err = clGetDeviceInfo ( deviceIds[deviceId], CL_DEVICE_TYPE, sizeof(infoType), &infoType, &infoStrLen );
            checkForError("clGetDeviceInfo[CL_DEVICE_TYPE]");
            OCLDeviceType deviceType;
            if (infoType & CL_DEVICE_TYPE_CPU)          deviceType = DEVICE_TYPE_CPU;
            if (infoType & CL_DEVICE_TYPE_GPU)          deviceType = DEVICE_TYPE_GPU;
            if (infoType & CL_DEVICE_TYPE_ACCELERATOR)  deviceType = DEVICE_TYPE_ACCELERATOR;
            if (infoType & CL_DEVICE_TYPE_DEFAULT)      deviceType = DEVICE_TYPE_DEFAULT;

            err = clGetDeviceInfo ( deviceIds[deviceId], CL_DEVICE_NAME, sizeof(infoStr), infoStr, &infoStrLen );
            checkForError("clGetDeviceInfo[CL_DEVICE_NAME]");
            QString deviceName = QString::fromLocal8Bit(infoStr, infoStrLen-1);

            err = clGetDeviceInfo ( deviceIds[deviceId], CL_DEVICE_VERSION, sizeof(infoStr), infoStr, &infoStrLen );
            checkForError("clGetDeviceInfo[CL_DEVICE_VERSION]");
            QString deviceVersion = QString::fromLocal8Bit(infoStr, infoStrLen-1);

            err = clGetDeviceInfo ( deviceIds[deviceId], CL_DEVICE_EXTENSIONS, sizeof(infoStr), infoStr, &infoStrLen );
            checkForError("clGetDeviceInfo[CL_DEVICE_EXTENSIONS]");
            QString deviceExtensions = QString::fromLocal8Bit(infoStr, infoStrLen-1);

            OpenCLDevice d(platformIds[platformId], platformName, deviceIds[deviceId],
                           deviceName, deviceType, deviceVersion, deviceExtensions);

            if(d.supportsDoublePrecision())
                devices << d;
            atLeastOneDeviceFound = true;

        } /* loop over devices */
    } /* loop over platforms */

    if(atLeastOneDeviceFound)
    {
        platformSupportet = true;
        currentDevice = OpenCLPlatformAndDeviceDialog::selectDevice(devices);

        cl_context_properties props[3];
        props[0] = CL_CONTEXT_PLATFORM;
        props[1] = (cl_context_properties)currentDevice.PlatformId;
        props[2] = 0;
        ctx = clCreateContext(props, 1, &currentDevice.DeviceId, NULL, NULL, &err);
        checkForError("clCreateContext");

        queue = clCreateCommandQueue(ctx, currentDevice.DeviceId, 0, &err);
        checkForError("clCreateCommandQueue");

        /* Setup clAmdBlas. */
        err = clAmdBlasSetup();
        checkForError("clAmdBlasSetup");

        /* Setup clAmdFft. */
        err = clAmdFftInitSetupData(&setupData);
        checkForError("clAmdFftInitSetupData");

        err = clAmdFftSetup(&setupData);
        checkForError("clAmdFftSetup");

        /* Custom Kernels */
        QFile kernels(":/kernels.cl");
        if (!kernels.open(QIODevice::ReadOnly))
        {
            qFatal("Error opening kernel file");
        }
        QByteArray code;
        while (!kernels.atEnd())
        {
            code.append(kernels.readLine());
        }
        const char* str = code.constData();
        kernels.close();
        cl_program program = clCreateProgramWithSource(ctx, 1, &str, NULL, &err);
        checkForError("clCreateProgramWithSource");
        err = clBuildProgram(program, 1, &currentDevice.DeviceId, NULL, NULL, NULL);
        char* buildLog;
        size_t buildLogSize;
        err = clGetProgramBuildInfo(program, currentDevice.DeviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
        checkForError("clGetProgramBuildInfo");
        buildLog = new char[buildLogSize+1];
        err = clGetProgramBuildInfo(program, currentDevice.DeviceId, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
        checkForError("clGetProgramBuildInfo");
        buildLog[buildLogSize] = '\0';
        qDebug() << "ocl build log:" << buildLog;
        complexAddKernel = clCreateKernel(program, "complex_add", &err);
        checkForError("clCreateKernel[complex_add]");
        complexMultKernel = clCreateKernel(program, "complex_mult", &err);
        checkForError("clCreateKernel[complex_mult]");
        complexScalarMultKernel = clCreateKernel(program, "complex_scalar_mult", &err);
        checkForError("clCreateKernel[complex_scalar_mult]");
        complexExpKernel = clCreateKernel(program, "complex_exp", &err);
        checkForError("clCreateKernel[complex_exp]");
        complexforMatricesKernel = clCreateKernel(program, "for_matrices", &err);
        checkForError("clCreateKernel[for_matrices]");
        complexFftshiftKernel = clCreateKernel(program, "complex_fftshift", &err);
        checkForError("clCreateKernel[for_matrices]");
    }
    else
    {
        platformSupportet = false;
    }

    // IMPORTANT:
    // we need to set the locale of the application for number formatting
    // back to the C standard as the call to QApplication() overwrites it
    // with the system locale. Apparently the number formatting c function is used
    // by clAmdFft to write the kernels on the fly. If the locale is set to german
    // for example, numbers are formattet with a comma as decimal delimiter
    // insted of a dot. This prodces an opencl compile error.
    setlocale(LC_NUMERIC,"C");
}

AppmlMath::~AppmlMath()
{
    clAmdBlasTeardown();
    if(plan)
    {
        err = clAmdFftDestroyPlan(&plan);
        checkForError("clAmdFftDestroyPlan");
    }
    err = clAmdFftTeardown();
    checkForError("clAmdFftTeardown");
    if(queue)
    {
        err = clReleaseCommandQueue(queue);
        checkForError("clReleaseCommandQueue");
    }
    if(ctx)
    {
        err = clReleaseContext(ctx);
        checkForError("clReleaseContext");
    }
}

void AppmlMath::checkForError(QString functionName)
{
    if(err != CL_SUCCESS)
    {
        if(queue)
            clReleaseCommandQueue(queue);
        if(ctx)
            clReleaseContext(ctx);
        QString error = "Error executing " + functionName + " (code " + QString::number(err) + ")";
        qFatal(error.toStdString().c_str());
    }
}

QString AppmlMath::getName()
{
    return "AMD APPML Math Plugin (Device: " + currentDevice.DeviceName + ")";
}

uint AppmlMath::getPerformanceHint()
{
    return 0;
//    if(currentDevice.DeviceType == DEVICE_TYPE_CPU)
//    {
//        return 40;
//    }
//    else
//    {
//        return 70;
//    }
}

bool AppmlMath::isPlatformSupported()
{
    return platformSupportet;
}

MathPlugin::DataType AppmlMath::getDataType()
{
    return DT_DOUBLE_PRECISION;
}

Matrix* AppmlMath::createMatrix(int row, int col)
{
   return new AppmlMatrix(ctx, queue, row, col);
}

Vector* AppmlMath::createVector(int s)
{
    return new AppmlVector(ctx, queue, s);
}

void AppmlMath::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
{
    cl_mem FX = static_cast<cl_mem>(fx->data());
    cl_mem FY = static_cast<cl_mem>(fy->data());
    int rows = fx->getRows();
    int cols = fx->getCols();
    err = clSetKernelArg(complexforMatricesKernel, 0, sizeof(cl_mem), &FX);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexforMatricesKernel, 1, sizeof(cl_mem), &FY);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexforMatricesKernel, 2, sizeof(cl_int), &rows);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexforMatricesKernel, 3, sizeof(cl_int), &cols);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexforMatricesKernel, 4, sizeof(cl_double), &stepx);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexforMatricesKernel, 5, sizeof(cl_double), &stepy);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {rows*cols};
    err = clEnqueueNDRangeKernel(queue, complexforMatricesKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(fx)->setDeviceDataChanged();
    static_cast<AppmlMatrix*>(fy)->setDeviceDataChanged();
}

void AppmlMath::mult(Matrix* a, Matrix* b, Matrix* c)
{
    DoubleComplex alpha = doubleComplex(1,0);
    DoubleComplex beta = doubleComplex(0,0);
    cl_mem A = static_cast<cl_mem>(a->data());
    cl_mem B = static_cast<cl_mem>(b->data());
    cl_mem C = static_cast<cl_mem>(c->data());

    err = clAmdBlasZgemm(clAmdBlasColumnMajor, clAmdBlasNoTrans, clAmdBlasNoTrans,
                         a->getRows(), b->getCols(), a->getCols(),
                         alpha, A, a->getRows(),
                         B, b->getRows(),
                         beta, C, c->getRows(),
                         1, &queue, 0, NULL, &event);
    checkForError("clAmdBlasZgemm");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clAmdBlasZgemm]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");

    // set data changed flag on c matrix
    static_cast<AppmlMatrix*>(c)->setDeviceDataChanged();
}

void AppmlMath::mult(Matrix* a, Vector* x, Vector* y)
{
    DoubleComplex alpha = doubleComplex(1,0);
    DoubleComplex beta = doubleComplex(0,0);
    cl_mem A = static_cast<cl_mem>(a->data());
    cl_mem X = static_cast<cl_mem>(x->data());
    cl_mem Y = static_cast<cl_mem>(y->data());

    err = clAmdBlasZgemv(clAmdBlasColumnMajor, clAmdBlasNoTrans,
            a->getRows(), a->getCols(),
            alpha, A, a->getRows(),
            X, 0, 1, beta, Y, 0, 1,
            1, &queue, 0, NULL, &event);
    checkForError("clAmdBlasZgemv");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clAmdBlasZgemv]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");

    // set data changed flag on c matrix
    static_cast<AppmlVector*>(y)->setDeviceDataChanged();
}

void AppmlMath::mult(Matrix *a, std::complex<double> s)
{    
    cl_mem A = static_cast<cl_mem>(a->data());
    err = clSetKernelArg(complexScalarMultKernel, 0, sizeof(cl_mem), &A);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexScalarMultKernel, 1, sizeof(std::complex<double>), &s);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {a->getRows()*a->getCols()};
    err = clEnqueueNDRangeKernel(queue, complexScalarMultKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(a)->setDeviceDataChanged();
}

void AppmlMath::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
{
    cl_mem A = static_cast<cl_mem>(a->data());
    cl_mem B = static_cast<cl_mem>(b->data());
    cl_mem C = static_cast<cl_mem>(c->data());
    err = clSetKernelArg(complexMultKernel, 0, sizeof(cl_mem), &A);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexMultKernel, 1, sizeof(cl_mem), &B);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexMultKernel, 2, sizeof(cl_mem), &C);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {a->getRows()*a->getCols()};
    err = clEnqueueNDRangeKernel(queue, complexMultKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(c)->setDeviceDataChanged();
}

void AppmlMath::componentWiseMult(Matrix* a, Matrix* b)
{
    componentWiseMult(a, b, a);
}

void AppmlMath::add(Matrix *a, Matrix *b)
{
    cl_mem A = static_cast<cl_mem>(a->data());
    cl_mem B = static_cast<cl_mem>(b->data());
    err = clSetKernelArg(complexAddKernel, 0, sizeof(cl_mem), &A);
    checkForError("clSetKernelArg");
    err = clSetKernelArg(complexAddKernel, 1, sizeof(cl_mem), &B);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {a->getRows()*a->getCols()};
    err = clEnqueueNDRangeKernel(queue, complexAddKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(a)->setDeviceDataChanged();
}

void AppmlMath::exp(Matrix *a)
{
    cl_mem A = static_cast<cl_mem>(a->data());
    err = clSetKernelArg(complexExpKernel, 0, sizeof(cl_mem), &A);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {a->getRows()*a->getCols()};
    err = clEnqueueNDRangeKernel(queue, complexExpKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(a)->setDeviceDataChanged();
}

void AppmlMath::pow(Matrix *a, int p)
{
    while(p > 1)
    {
        componentWiseMult(a, a);
        p--;
    }
}

void AppmlMath::fft(Matrix *a, clAmdFftDirection dir)
{
    if(currentFftPlanSize != QSize(a->getRows(), a->getCols()))
    {
        if(plan)
        {
            err = clAmdFftDestroyPlan(&plan);
            checkForError("clAmdFftDestroyPlan");
            plan = 0;
        }

        const size_t dims[2] = {a->getRows(), a->getCols()};
        err = clAmdFftCreateDefaultPlan(&plan, ctx, CLFFT_2D, dims);
        if(err == CLFFT_NOTIMPLEMENTED)
        {
            qDebug() << ("clAmdFftCreateDefaultPlan for " + QString::number(a->getRows()) +
                         " rows and " + QString::number(a->getCols()) + " cols not implemented jet");
            return;
        }
        else
        {
            checkForError("clAmdFftCreateDefaultPlan");
        }

        err = clAmdFftSetPlanPrecision(plan, CLFFT_DOUBLE);
        checkForError("clAmdFftSetPlanPrecision");

        //      0 is a valid value for numQueues, in
        //      which case client does not want the runtime to run load experiments
        //      and only pre-calculate state information
        err = clAmdFftBakePlan(plan, 0, &queue, NULL, NULL);
        checkForError("clAmdFftBakePlan");

        // for some reason we have to create a new plan every time,
        // otherwise we get an error on the second execution of
        // clAmdFftEnqueueTransform for the same plan
        //currentFftPlanSize = QSize(a->getRows(), a->getCols());
    }
    cl_mem A = static_cast<cl_mem>(a->data());
    err = clAmdFftEnqueueTransform(plan, dir, 1, &queue, 0, NULL, NULL, &A, NULL, NULL);
    checkForError("clAmdFftEnqueueTransform");

    err = clFinish(queue);
    checkForError("clFinish");

//    err = clAmdFftEnqueueTransform(plan, dir, 1, &queue, 0, NULL, &event, &A, NULL, NULL);
//    checkForError("clAmdFftEnqueueTransform");

//    err = clWaitForEvents(1, &event);
//    checkForError("clWaitForEvents[clAmdFftEnqueueTransform]");
//    err = clReleaseEvent(event);
//    checkForError("clReleaseEvent");

    // set data changed flag on a matrix
    static_cast<AppmlMatrix*>(a)->setDeviceDataChanged();
}

void AppmlMath::fft(Matrix *a)
{
    fft(a, CLFFT_FORWARD);
}

void AppmlMath::ifft(Matrix *a)
{
    fft(a, CLFFT_BACKWARD);
}

void AppmlMath::fft(Matrix *in, Matrix *out)
{
}

void AppmlMath::ifft(Matrix *in, Matrix *out)
{
}

void AppmlMath::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols() % 2 == 0);

    cl_mem A = static_cast<cl_mem>(a->data());
    err = clSetKernelArg(complexFftshiftKernel, 0, sizeof(cl_mem), &A);
    checkForError("clSetKernelArg");
    int rows = a->getRows();
    err = clSetKernelArg(complexFftshiftKernel, 1, sizeof(cl_int), &rows);
    checkForError("clSetKernelArg");
    int cols = a->getCols();
    err = clSetKernelArg(complexFftshiftKernel, 2, sizeof(cl_int), &cols);
    checkForError("clSetKernelArg");
    size_t workSize[1] = {a->getRows()*a->getCols()/2};
    err = clEnqueueNDRangeKernel(queue, complexFftshiftKernel, 1, NULL, workSize, NULL, 0, NULL, &event);
    checkForError("clEnqueueNDRangeKernel");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueNDRangeKernel]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    static_cast<AppmlMatrix*>(a)->setDeviceDataChanged();
}

void AppmlMath::ifftshift(Matrix *a)
{
    fftshift(a);
}

Q_EXPORT_PLUGIN2(hipnosappmlmathplugin, AppmlMath);
