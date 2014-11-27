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

#ifndef VTKPIPELINE_H
#define VTKPIPELINE_H

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkImageData.h>
#include <vtkWarpScalar.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkCastToConcrete.h>
#include <vtkPolyDataNormals.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkCubeAxesActor.h>
#include <vtkLogoRepresentation.h>
#include <vtkQImageToImageSource.h>
#include <vtkProperty2D.h>
#include <vtkColorTransferFunction.h>
#include <vtkWindowToImageFilter.h>
#include <vtkJPEGWriter.h>
#include <vtkGenericOpenGLRenderWindow.h>

#include <QMutex>

#include "vtkviewpropertywidget.h"
#include "common/pipelinedata.h"

/**
 * @brief Class that wraps the VTK pipeline and connects it to the rest of the application.
 *            By hiding VTK behind this class we could replace it with other rendering techniques
 */
class VtkPipeline : public QObject
{
    Q_OBJECT

public:
    VtkPipeline(vtkRenderWindow* rw);

    void update(PipelineSpectrumData* spectrum);
    void updateAxes();
    void exportAsJpeg(QString filePath);
    void setResetCamera();

    // Data struktures
    vtkSmartPointer<vtkFloatArray> realData;  
    vtkSmartPointer<vtkFloatArray> imagData;  
    vtkSmartPointer<vtkFloatArray> absData;  
    vtkSmartPointer<vtkFloatArray> argData;  
    vtkSmartPointer<vtkImageData> imgData;  

signals:
    void dataBoundsChanged(double, double);
    void spectrumChanged(Spectrum s);

public slots:
    void render();
    void setWarpingScaling(double s);
    void setColorTransferFunction(QGradient grad, float min, float max);
    void setWarpingMapping(VtkViewPropertyWidget::MappingFunction f);
    void setColorMapping(VtkViewPropertyWidget::MappingFunction f);
    void setFrequencyPercentageInterval(int min, int max);

private:
    Matrix* getComplexAmplitudeFromSpectrum(PipelineSpectrumData *s);

    QMutex vtkMutex;  
    bool resetCamera;  
    QString currentColorMappint;  
    vtkRenderWindow* window;  
    int frequencyMinPercentage;
    int frequencyMaxPercentage;

    // Filters
    vtkSmartPointer<vtkImageDataGeometryFilter> geomFilter;  
    vtkSmartPointer<vtkWarpScalar> warp;  
    vtkSmartPointer<vtkCastToConcrete> caster;  
    vtkSmartPointer<vtkPolyDataNormals> normals;  
    vtkSmartPointer<vtkPolyDataMapper> mapper;  
    vtkSmartPointer<vtkActor> actor;  
    vtkSmartPointer<vtkCubeAxesActor> cubeAxesActor;  
    vtkSmartPointer<vtkRenderer> renderer;  
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction;  
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter;  
    vtkSmartPointer<vtkJPEGWriter> jpegWriter;  
};

#endif // VTKPIPELINE_H
