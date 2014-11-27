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

#include "csvsourcecomponent.h"
#include "common/math/math.h"
#include <QDebug>
#include <QFile>

CsvSourceComponent::CsvSourceComponent(QString iDataPath, QString pDataPath, QString sep, Spectrum s, double step)
    : PipelineComponent()
{
    setNumberOfInputConnections(0);
    setNumberOfOutputConnections(1);
    intensityDataPath = iDataPath;
    phaseDataPath = pDataPath;
    spectrum = s;
    stepSize = step;
    csvSeparator = sep;
}

CsvSourceComponent::~CsvSourceComponent()
{
}

QString CsvSourceComponent::getType()
{
    return "CSV beam source";
}

QIcon CsvSourceComponent::getIcon()
{
    return QIcon(":/icons/components/csv-source.png");
}

PipelineComponent* CsvSourceComponent::clone()
{
    return new CsvSourceComponent(intensityDataPath, phaseDataPath, csvSeparator, spectrum, stepSize);
}

void CsvSourceComponent::setProperty(Property p)
{
    if(p.getName() == "Step size" && p.getType() == Property::PT_DOUBLE)
    {
        stepSize = p.getDoubleValue();
    }
    else if(p.getName() == "Intensity" && p.getType() == Property::PT_FILE)
    {
        intensityDataPath = p.getFileName();
    }
    else if(p.getName() == "Phase" && p.getType() == Property::PT_FILE)
    {
        phaseDataPath = p.getFileName();
    }
    else if(p.getName() == "CSV separator" && p.getType() == Property::PT_STRING)
    {
        csvSeparator = p.getStringValue();
    }
    else if(p.getName() == "Spectrum" && p.getType() == Property::PT_SPECTRUM)
    {
        spectrum = p.getSpectrum();
    }
}

QList<PipelineComponent::Property> CsvSourceComponent::getProperties()
{
    QList<Property> l;
    l << Property(this, "Spectrum", "Sets the spectrum", spectrum);
    l << Property(this, "Step size", "Specifies the step size", stepSize, 0.0,  10000.0, "m");
    l << Property(this, "Intensity", "Holds the path to the CSV file containing the intensity distribution. Can NOT be empty!", QFile(intensityDataPath));
    l << Property(this, "Phase", "Holds the path to the CSV file containing  the phase distribution. Can be empty!", QFile(phaseDataPath));
    l << Property(this, "CSV separator", "Specifies the CSV separator", csvSeparator);
    return l;
}

void CsvSourceComponent::gaussPropagation(double z)
{
    int resolution = 0;
    QFile intensityFile(intensityDataPath);
    if (!intensityFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Could not open file " + intensityDataPath;
        resolution = 256;
    }
    else
    {
        while (!intensityFile.atEnd())
        {
            intensityFile.readLine();
            resolution++;
        }
    }
    intensityFile.close();
    double w = stepSize * resolution  / std::sqrt(M_PI * resolution);
    getOutputConnection()->setData(
                PipelineSpectrumData::createGaussSpectrumData(spectrum, w, 0.0, 1.0, resolution));
}

void CsvSourceComponent::fourierPropagation(double z)
{
    int resolution = 256;
    QFile intensityFile(intensityDataPath);
    if (intensityDataPath.isNull() || intensityDataPath.isEmpty() ||
            !intensityFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Could not open file " + intensityDataPath;
        getOutputConnection()->setData(
                    PipelineSpectrumData::createFourierSpectrumData(spectrum, resolution, stepSize));
        return;
    }
    QList<QString> intensityLines;
    int rows = 0;
    while (!intensityFile.atEnd())
    {
        intensityLines << intensityFile.readLine();
        rows++;
    }
    intensityFile.close();

    QFile phaseFile(phaseDataPath);
    QList<QString> phaseLines;
    if (phaseDataPath.isNull() || phaseDataPath.isEmpty() ||
            phaseFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        while (!phaseFile.atEnd())
        {
            phaseLines << phaseFile.readLine();
        }
    }
    phaseFile.close();

    int cols = intensityLines.first().split(",").size();
    resolution = qMin(rows, cols);
    PipelineSpectrumData* spectrumData =
            PipelineSpectrumData::createFourierSpectrumData(spectrum, resolution, stepSize);
    for(int k = 0; k < spectrum.size(); k++)
    {
        FourierPipelineData* data = spectrumData->getData<FourierPipelineData>(k);
        for(int i = 0; i < resolution; i++)
        {
            QStringList intVals = intensityLines[i].split(",");
            QStringList phaseVals;
            if(i < phaseLines.size())
                phaseVals = phaseLines[i].split(csvSeparator);
            for(int j = 0; j < resolution; j++)
            {
                double intensity = 0;
                if(j < intVals.size())
                    intensity = intVals.at(j).toDouble();
                double phase = 0;
                if(j < phaseVals.size())
                    phase = phaseVals.at(j).toDouble();
                double abs = std::sqrt(intensity);
                data->ComplexAmplitude->set(i, j, std::complex<double>(abs*std::cos(phase), abs*std::sin(phase)));
            }
        }
    }
    getOutputConnection()->setData(spectrumData);
}
