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

#include "spectrum.h"
#include <cmath>
#include <QDebug>

/**
 * @brief
 *
 */
Spectrum::Spectrum()
{
    sorted = false;
}

/**
 * @brief
 *
 * @param l
 */
Spectrum::Spectrum(double f)
{
    addEntry(f, 1);
}

/**
 * @brief
 *
 */
Spectrum::~Spectrum()
{

}

/**
 * @brief
 *
 * @param f
 * @param i
 */
void Spectrum::addEntry(double f, double i)
{
    addEntry(SpectrumEntry(f, i));
}

/**
 * @brief
 *
 * @param e
 */
void Spectrum::addEntry(SpectrumEntry e)
{
    entries << e;
    sorted = false;
}

/**
 * @brief
 *
 * @return double
 */
double Spectrum::getNormalizationFactor()
{
    if(entries.size() == 0)
        return 0.0;
    if(entries.size() == 1)
        return entries.first().Intensity;
    double sum = 0;
    for (int i = 0; i < entries.size(); i++)
    {
        sum += getStepWidth(i) * getEntry(i).Intensity;
    }
    return sum;
}

double Spectrum::getStepWidth(int i)
{
    if(entries.size() == 0)
        return 0.0;
    if(entries.size() == 1)
        return 1.0;
    if(!sorted)
        sort();
    double width = 0;
    if(i > 0)
    {
        width += (entries[i].Frequency - entries[i-1].Frequency) / 2.0;
    }
    if(i < size()-1)
    {
        width += (entries[i+1].Frequency - entries[i].Frequency) / 2.0;
    }
    return width;
}

/**
 * @brief
 *
 * @return QVector<Spectrum::SpectrumEntry>
 */
QVector<Spectrum::SpectrumEntry>& Spectrum::getEntries()
{
    if(!sorted)
        sort();
    return entries;
}

/**
 * @brief
 *
 * @param i
 * @return Spectrum::SpectrumEntry &
 */
Spectrum::SpectrumEntry& Spectrum::getEntry(int i)
{
    if(!sorted)
        sort();
    return entries[i];
}

/**
 * @brief
 *
 */
void Spectrum::sort()
{
    qSort(entries.begin(), entries.end(), SpectrumEntry::LessThan());
    sorted = true;
}

/**
 * @brief
 *
 * @return int
 */
int Spectrum::size()
{
    return entries.size();
}

/**
 * @brief
 *
 * @param spectrum
 * @param resolution
 * @param center
 * @param radius
 */
void Spectrum::generateGaussianSpectrum(Spectrum *spectrum, int resolution, double center, double radius)
{
    spectrum->getEntries().resize(0);
    double step = radius / double(resolution) * 6.0;
    double offset = double(resolution)/2.0 * step - 0.5 * step;
    for(int i = 0; i < resolution; i++)
    {
        double freq = double(i) * step + center - offset;
        if(freq > 0)
        {
            double x = freq - center;
            spectrum->addEntry(freq, std::exp(-x*x/(radius*radius)));
        }
    }
}

/**
 * @brief
 *
 * @param spectrum
 * @param resolution
 * @param center
 * @param width
 */
void Spectrum::generateBlockSpectrum(Spectrum *spectrum, int resolution, double center, double width)
{
    spectrum->getEntries().resize(0);
    double step = width / double(resolution);
    double offset = double(resolution)/2.0 * step - 0.5 * step;
    double I = 1.0/width;
    for(int i = 0; i < resolution; i++)
    {
        double freq = double(i) * step + center - offset;
        if(freq > 0)
        {
            spectrum->addEntry(freq, I);
        }
    }
}

/**
 * @brief
 *
 * @param spectrum
 * @param resolution
 * @param center
 * @param width
 */
void Spectrum::generateSechSpectrum(Spectrum *spectrum, int resolution, double center, double width)
{
    spectrum->getEntries().resize(0);
    double step = width / double(resolution);
    double offset = double(resolution)/2.0 * step - 0.5 * step;
    for(int i = 0; i < resolution; i++)
    {
        double freq = double(i) * step + center - offset;
        if(freq > 0)
        {
            double x = (freq - center) * 20 / width; // we let sech run from -10 to 10
            spectrum->addEntry(freq, 1.0 / std::cosh(x));
        }
    }
}
