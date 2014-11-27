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

#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <QVector>
#include <QMetaType>

/**
 * @brief This class represents a spectrum. Basicly a list of frequency and intesity pairs
 *
 */
class Spectrum
{
public:

    /**
     * @brief A frequency and intesity pair
     *
     */
    struct SpectrumEntry
    {
        /**
         * @brief Struct used to sort SpectrumEntry by frequency
         *
         */
        struct LessThan
        {
            bool operator()(SpectrumEntry a, SpectrumEntry b) const
            {
                return (a.Frequency < b.Frequency);
            }
        };

        /**
         * @brief Constructs a new SpectrumEntry
         *
         * @param f Frequency
         * @param i Intensity
         */
        SpectrumEntry(double f = 0, double i = 0)
        {
            Frequency = f;
            Intensity = i;
        }

        double Frequency;
        double Intensity;
    };

    Spectrum();
    Spectrum(double f);
    ~Spectrum();

    static void generateGaussianSpectrum(Spectrum* spectrum, int resolution, double center, double radius);
    static void generateBlockSpectrum(Spectrum *spectrum, int resolution, double center, double width);
    static void generateSechSpectrum(Spectrum *spectrum, int resolution, double center, double width);

    void addEntry(double f, double i);
    void addEntry(SpectrumEntry e);
    double getNormalizationFactor();
    double getStepWidth(int i);
    int size();
    void sort();
    QVector<SpectrumEntry> &getEntries();
    SpectrumEntry& getEntry(int i);

private:
    QVector<SpectrumEntry> entries;
    bool sorted;

};
Q_DECLARE_METATYPE(Spectrum)
#endif // SPECTRUM_H
