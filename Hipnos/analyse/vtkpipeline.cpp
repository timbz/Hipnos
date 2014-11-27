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

#include "vtkpipeline.h"

#include <QImage>

VtkPipeline::VtkPipeline(vtkRenderWindow *rw)
{
    window = rw;
    resetCamera = true;
    currentColorMappint = "real";
    frequencyMinPercentage = 0;
    frequencyMaxPercentage = 100;

    // Create scalar data to associate with the vertices of the plane
    // Real component of E
    realData = vtkSmartPointer<vtkFloatArray>::New();
    realData->SetName("real");

    // Imag component of E
    imagData = vtkSmartPointer<vtkFloatArray>::New();
    imagData->SetName("imag");

    // Square absolute value of E
    absData = vtkSmartPointer<vtkFloatArray>::New();
    absData->SetName("abs");

    // Arg of E
    argData = vtkSmartPointer<vtkFloatArray>::New();
    argData->SetName("arg");

    // Image input data
    imgData = vtkSmartPointer<vtkImageData>::New();
    imgData->GetPointData()->SetScalars(realData);
    imgData->GetPointData()->AddArray(imagData);
    imgData->GetPointData()->AddArray(absData);
    imgData->GetPointData()->AddArray(argData);

    // Image data to geometry filter
    geomFilter =vtkSmartPointer<vtkImageDataGeometryFilter>::New();
    geomFilter->SetInput(imgData);

    // Warp
    warp = vtkSmartPointer<vtkWarpScalar>::New();
    warp->SetInput(geomFilter->GetOutput());
    warp->UseNormalOn();
    warp->SetNormal(0.0,0.0,1.0);

    // Cast point set to PolyData
    caster = vtkSmartPointer<vtkCastToConcrete>::New();
    caster->SetInput(warp->GetOutput());

    // Recalculate normals (because of warping)
    normals = vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInput(caster->GetPolyDataOutput());
    normals->SetFeatureAngle(60);

    // Mapper
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    // From the documentation:
    // Immediate mode rendering tends to be slower but it can handle larger datasets.
    // The default value is immediate mode off. If you are having problems rendering
    // a large dataset you might want to consider using immediate more rendering.
    //mapper->ImmediateModeRenderingOn();
    mapper->SetInput(normals->GetOutput());
    mapper->ScalarVisibilityOn();
    mapper->SetColorModeToMapScalars();
    mapper->SetScalarModeToUsePointFieldData();

    // Color lookup table
    colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();

    mapper->SetLookupTable(colorTransferFunction);

    // Actor in scene
    actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // VTK Renderer
    renderer = vtkSmartPointer<vtkRenderer>::New();

    // Add Actor to renderer
    renderer->AddActor(actor);

    // Axis
    cubeAxesActor = vtkSmartPointer<vtkCubeAxesActor>::New();
    cubeAxesActor->SetCamera(renderer->GetActiveCamera());
    cubeAxesActor->SetFlyModeToStaticTriad();
    cubeAxesActor->SetXTitle("x");
    cubeAxesActor->SetXUnits("m");
    cubeAxesActor->SetYTitle("y");
    cubeAxesActor->SetYUnits("m");

    renderer->AddActor(cubeAxesActor);
    renderer->ResetCamera();

    // VTK/Qt wedded
    window->AddRenderer(renderer);

    // Window to image
    windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(window);

    // JPEG Writer
    jpegWriter = vtkSmartPointer<vtkJPEGWriter>::New();
    jpegWriter->SetInputConnection(windowToImageFilter->GetOutputPort());
}

Matrix* VtkPipeline::getComplexAmplitudeFromSpectrum(PipelineSpectrumData* s)
{
    Spectrum sectrum = s->getSpectrum();
    double step = (sectrum.getEntries().last().Frequency - sectrum.getEntries().first().Frequency) / 100.0;
    double min = sectrum.getEntries().first().Frequency + step * frequencyMinPercentage;
    double max = sectrum.getEntries().first().Frequency + step * frequencyMaxPercentage;
    return s->getComplexAmplitude(min, max);
}

void VtkPipeline::update(PipelineSpectrumData* spectrum)
{
    vtkMutex.lock();

        Matrix* data = getComplexAmplitudeFromSpectrum(spectrum);

        if(data != 0)
        {
            emit spectrumChanged(spectrum->getSpectrum());
            // Calc and set bounds, center and resolution
            double samplingStepSize = spectrum->getSamplingStepSize();
            imgData->SetExtent(0, data->getCols()-1, 0, data->getRows()-1, 0, 0);
            double offsetx = - data->getCols()/2 * samplingStepSize;
            double offsety = - data->getRows()/2 * samplingStepSize;
            imgData->SetOrigin(offsetx, offsety, 0.0);
            imgData->SetSpacing(samplingStepSize, samplingStepSize, samplingStepSize);

            int size = data->getCols() * data->getRows();
            realData->SetNumberOfValues(size);
            imagData->SetNumberOfValues(size);
            absData->SetNumberOfValues(size);
            argData->SetNumberOfValues(size);

            // Fill with data
            int k = 0;
            for(int i = data->getRows()-1; i >= 0; i--)
            {
                for(int j = 0; j < data->getCols(); j++)
                {
                    std::complex<double> tmp = data->get(i,j);
                    realData->GetPointer(0)[k] = std::real(tmp);
                    imagData->GetPointer(0)[k] = std::imag(tmp);
                    argData->GetPointer(0)[k] = std::arg(tmp);
                    double a = std::abs(tmp);
                    absData->GetPointer(0)[k] = a*a;
                    k++;
                }
            }

            realData->Modified();
            imagData->Modified();
            absData->Modified();
            argData->Modified();
            imagData->Modified();
            mapper->Modified();
            mapper->Update();
            updateAxes();
        }

    vtkMutex.unlock();
}

void VtkPipeline::render()
{
    // We need to sync this with the update.
    // If we try to render while the worker thread
    // calls updates, VTK crashes.
    // Insted of locking we use tryLock.
    // If the lock fails we skip rendering. The thread
    // that is holding the lock will trigger a callback
    // that renders the pipeline (see AnalysisController::onThreadFinished())
    if(vtkMutex.tryLock())
    {
        if(resetCamera)
        {
            renderer->ResetCamera();
            //renderer->ResetCameraClippingRange();
            resetCamera = false;
        }
        double* range = imgData->GetPointData()->GetArray(currentColorMappint.toLocal8Bit().data())->GetRange();
        emit dataBoundsChanged(range[0], range[1]);
        window->Render(); // on win with resolution > 1000 this call crashes
        vtkMutex.unlock();
    }
}

void VtkPipeline::exportAsJpeg(QString filePath)
{
    if(!filePath.isEmpty() && !filePath.isNull())
    {
        if(!filePath.endsWith(".jpg"))
            filePath += ".jpg";
        // Same as renderVtkPipeline()
        // Dont render while the worker thread is running
        vtkMutex.lock();
            windowToImageFilter->Modified();
            windowToImageFilter->Update();
            jpegWriter->SetFileName(filePath.toAscii());
            jpegWriter->Write();
        vtkMutex.unlock();
    }
}

void VtkPipeline::updateAxes()
{
    double* bounds = mapper->GetInput()->GetBounds();
    // set z min to 0
    bounds[4] = 0;
    cubeAxesActor->SetZAxisRange(0, bounds[5]/warp->GetScaleFactor());
    cubeAxesActor->SetBounds(bounds);
    // scale z axis
}

void VtkPipeline::setResetCamera()
{
    resetCamera = true;
}

void VtkPipeline::setColorTransferFunction(QGradient grad, float min, float max)
{
    colorTransferFunction->RemoveAllPoints();
    foreach(QGradientStop stop, grad.stops())
    {
        // QGradient values go from 0 to 1 so we map them to the min max range
        float value = min + stop.first * (max - min);
        colorTransferFunction->AddRGBPoint(value, stop.second.redF(), stop.second.greenF(), stop.second.blueF());
    }
}

void VtkPipeline::setColorMapping(VtkViewPropertyWidget::MappingFunction f)
{
    currentColorMappint = f.Key;
    mapper->SelectColorArray(f.Key.toStdString().c_str());
}

void VtkPipeline::setFrequencyPercentageInterval(int min, int max)
{
    frequencyMinPercentage = min;
    frequencyMaxPercentage = max;
}

void VtkPipeline::setWarpingMapping(VtkViewPropertyWidget::MappingFunction f)
{
    warp->SetInputArrayToProcess(0,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS, f.Key.toStdString().c_str());
    cubeAxesActor->SetZTitle(f.Name.toLocal8Bit().data());
    cubeAxesActor->SetZUnits(f.Unit.toLocal8Bit().data());

    mapper->Update();
    updateAxes();
    setResetCamera();
}

void VtkPipeline::setWarpingScaling(double s)
{
    warp->SetScaleFactor(s);
    mapper->Update();
    updateAxes();
    setResetCamera();
}
