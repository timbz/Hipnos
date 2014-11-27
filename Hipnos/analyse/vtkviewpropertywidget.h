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

#ifndef VTKVIEWPROPERTYWIDGET_H
#define VTKVIEWPROPERTYWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QCheckBox>
#include <QDoubleValidator>
#include <QLabel>

#include "common/widgets/gradienteditor.h"
#include "common/widgets/qtcolortriangle.h"
#include "common/widgets/qxtspanslider.h"
#include "common/spectrum.h"


/**
 * @brief QWidget that displays properties used to visualise the data with VTK (like the warp scaling and the color map).
 *
 */
class VtkViewPropertyWidget : public QWidget
{
    Q_OBJECT
public:

    /**
     * @brief Struct that acts a key to a function used to visualise complex data (for example std::real(), std::imag(), std::arg() and std::abs()))
     *
     */
    struct MappingFunction
    {
        QString Key;   /**< The key used iternaly to describe this function */
        QString Name;   /**< The name of the function that gets displayed in the GUI */
        QString Unit;   /**< A QString containing the unit of the result */

        /**
         * @brief Creates a new VtkViewPropertyWidget::MappingFunction
         *
         * @param k
         * @param n
         * @param u
         */
        MappingFunction(QString k = "", QString n = "", QString u = "")
        {
            Key = k;
            Name = n;
            Unit = u;
        }
    };

    explicit VtkViewPropertyWidget(QWidget *parent = 0);

    QGradient getGradient();
    MappingFunction getWarpingMapping();
    MappingFunction getColorMapping();
    float getGradientMinValue();

    float getGradientMaxValue();
    double getWarpingScaling();
    int getFrequencyMinPercentage();
    int getFrequencyMaxPercentage();

signals:
    void gradientChanged(QGradient g, float min, float max);
    void warpingMappingChanged(VtkViewPropertyWidget::MappingFunction f);
    void gradientMappingChanged(VtkViewPropertyWidget::MappingFunction f);
    void warpingScalingChanged(double i);
    void frequencyPercentageRangeChanged(int min, int max);
    void applyProperties();

public slots:
    void setGradientMinMax(double min, double max);
    void setSpectrum(Spectrum s);
    void updateFrequencyMinMax(int lower, int upper);

private slots:
    void onGradientWidgetChanged(QGradient g);
    void onGradientMinMaxValueChanged(QString text);
    void onWarpingCompboBoxChanged(int i);
    void onGradientCompboBoxChanged(int i);
    void onWarpingScalingChanged(double i);
    void onApplyClicked();
    void onGradientAutoMinMaxChanged(int state);

private:
    void addSpacer(QString text);
    QComboBox* createMappinfFunctionComboBox();

    bool warpingMappingChangedFlag;  
    bool colorMappingChangedFlag;  
    bool gradientChangedFlag;  
    bool warpingScalingChangedFlag;  
    bool frequencyMinMaxChangedFlag;  

    QDoubleValidator doubleValidator;  
    GradientEditor* gradientEditor;  
    QtColorTriangle* colorTriangle;  
    QLineEdit* gradientMinValue;  
    QLineEdit* gradientMaxValue;  
    QCheckBox* autoMinMaxValue;  
    QxtSpanSlider* frequency;  
    QLabel* frequencyMinLabel;  
    QLabel* frequencyMaxLabel;  
    double frequencyMin;  
    double frequencyMax;  
    double spectrumMinFrequency;  
    double spectrumMaxFrequency;  

    QFont spacerFont;  
    QComboBox* warpingCompboBox;  
    QComboBox* gradientCompboBox;  
    QPushButton* applyButton;  
    QDoubleSpinBox* warpingScaling;  
    QHash<QString, MappingFunction> mappingFunctions;  
};

#endif // VTKVIEWPROPERTYWIDGET_H
