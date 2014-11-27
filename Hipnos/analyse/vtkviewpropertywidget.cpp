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

#include "vtkviewpropertywidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <QFrame>
#include <QDebug>

VtkViewPropertyWidget::VtkViewPropertyWidget(QWidget *parent) :
    QWidget(parent)
{
    warpingMappingChangedFlag = false;
    colorMappingChangedFlag = false;
    gradientChangedFlag = false;
    warpingScalingChangedFlag = false;
    frequencyMinMaxChangedFlag = false;

    mappingFunctions.insert("real", MappingFunction("real", "Real{E}", "m"));
    mappingFunctions.insert("imag", MappingFunction("imag", "Imag{E}", "s"));
    mappingFunctions.insert("abs", MappingFunction("abs", trUtf8("Abs{E}²"), "g"));
    mappingFunctions.insert("arg", MappingFunction("arg", "Arg{E}", "t"));

    QVBoxLayout* layout = new QVBoxLayout(this);

    spacerFont.setBold(true);

    addSpacer("Spectrum");
    layout->addSpacing(5);

    frequency = new QxtSpanSlider(Qt::Horizontal);
    frequency->setFocusPolicy(Qt::NoFocus);
    layout->addWidget(frequency);
    frequency->setSingleStep(1);
    frequency->setRange(0, 100);
    frequency->setSpan(0, 100);
    QHBoxLayout* frequencyMinMaxLayout = new QHBoxLayout;
    layout->addLayout(frequencyMinMaxLayout);
    frequencyMinLabel = new QLabel;
    frequencyMinLabel->setStyleSheet("font: italic;");
    frequencyMaxLabel = new QLabel;
    frequencyMaxLabel->setStyleSheet("font: italic;");
    frequencyMaxLabel->setAlignment(Qt::AlignRight);
    frequencyMinMaxLayout->addWidget(frequencyMinLabel);
    frequencyMinMaxLayout->addWidget(frequencyMaxLabel);
    connect(frequency, SIGNAL(spanChanged(int,int)), this, SLOT(updateFrequencyMinMax(int,int)));

    layout->addSpacing(20);

    addSpacer("Warping");

    layout->addSpacing(5);

    /* ------ Warping Editor ------ */

    QFrame* warpingEditorFrame = new QFrame();
    layout->addWidget(warpingEditorFrame);
    QVBoxLayout* warpingEditorFrameLayout = new QVBoxLayout(warpingEditorFrame);
    warpingEditorFrameLayout->setSpacing(0);
    warpingEditorFrameLayout->setContentsMargins(0, 0, 0, 0);

    warpingCompboBox = createMappinfFunctionComboBox();
    warpingEditorFrameLayout->addWidget(warpingCompboBox);
    connect(warpingCompboBox, SIGNAL(activated(int)), this, SLOT(onWarpingCompboBoxChanged(int)));

    warpingEditorFrameLayout->addSpacing(20);

    warpingScaling = new QDoubleSpinBox();
    warpingEditorFrameLayout->addWidget(warpingScaling);
    warpingScaling->setValue(0);
    warpingScaling->setRange(0, 100);
    warpingScaling->setDecimals(4);
    warpingScaling->setSingleStep(0.01);
    warpingScaling->setPrefix("Warping scaling: ");
    connect(warpingScaling, SIGNAL(valueChanged(double)), this, SLOT(onWarpingScalingChanged(double)));

    /* ------ Warping Editor end ------ */

    layout->addSpacing(20);
    addSpacer("Color Mapping");
    layout->addSpacing(5);

    /* ------ Gradient Editor ------ */

    QFrame* gradientEditorFrame = new QFrame();
    layout->addWidget(gradientEditorFrame);
    QVBoxLayout* gradientEditorFrameLayout = new QVBoxLayout(gradientEditorFrame);
    gradientEditorFrameLayout->setSpacing(0);
    gradientEditorFrameLayout->setContentsMargins(0, 0, 0, 0);

    gradientCompboBox = createMappinfFunctionComboBox();
    gradientEditorFrameLayout->addWidget(gradientCompboBox);
    connect(gradientCompboBox, SIGNAL(activated(int)), this, SLOT(onGradientCompboBoxChanged(int)));
    gradientEditorFrameLayout->addSpacing(20);

    QFrame* gradientMinMaxFrame = new QFrame();
    gradientEditorFrameLayout->addWidget(gradientMinMaxFrame);
    QHBoxLayout* gradientMinMaxFrameLayout = new QHBoxLayout(gradientMinMaxFrame);
    gradientMinValue = new QLineEdit();
    gradientMinValue->setMaximumWidth(80);
    gradientMinValue->setText("0");
    gradientMinValue->setValidator(&doubleValidator);
    connect(gradientMinValue, SIGNAL(textEdited(QString)), this, SLOT(onGradientMinMaxValueChanged(QString)));
    gradientMaxValue = new QLineEdit();
    gradientMaxValue->setText("1");
    gradientMaxValue->setMaximumWidth(80);
    gradientMaxValue->setValidator(&doubleValidator);
    connect(gradientMaxValue, SIGNAL(textEdited(QString)), this, SLOT(onGradientMinMaxValueChanged(QString)));

    QLabel* minLabel = new QLabel();
    minLabel->setText("Min:");
    QLabel* maxLabel = new QLabel();
    maxLabel->setText("Max:");

    autoMinMaxValue = new QCheckBox();
    autoMinMaxValue->setToolTip("Manual Min/Max");
    autoMinMaxValue->setChecked(false);
    gradientMinValue->setDisabled(true);
    gradientMaxValue->setDisabled(true);
    gradientMinMaxFrameLayout->addWidget(autoMinMaxValue);
    connect(autoMinMaxValue, SIGNAL(stateChanged(int)), this, SLOT(onGradientAutoMinMaxChanged(int)));

    gradientMinMaxFrameLayout->addWidget(minLabel);
    gradientMinMaxFrameLayout->addWidget(gradientMinValue);
    gradientMinMaxFrameLayout->addItem(new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum));
    gradientMinMaxFrameLayout->addWidget(maxLabel);
    gradientMinMaxFrameLayout->addWidget(gradientMaxValue);

    QLinearGradient defaultGradient;
    defaultGradient.setColorAt(0, Qt::yellow);
    defaultGradient.setColorAt(0.2, Qt::green);
    defaultGradient.setColorAt(0.4, Qt::cyan);
    defaultGradient.setColorAt(0.6, Qt::blue);
    defaultGradient.setColorAt(0.8, Qt::magenta);
    defaultGradient.setColorAt(1, Qt::red);

    gradientEditor = new GradientEditor();
    gradientEditor->setMinimumHeight(30);
    gradientEditor->setMaximumHeight(30);
    gradientEditorFrameLayout->addWidget(gradientEditor);
    gradientEditor->setGradient(defaultGradient);

    colorTriangle = new QtColorTriangle();
    colorTriangle->setMinimumHeight(150);
    colorTriangle->setMaximumHeight(150);
    gradientEditorFrameLayout->addSpacing(10);
    gradientEditorFrameLayout->addWidget(colorTriangle);
    colorTriangle->setColor(gradientEditor->color());

    connect(gradientEditor, SIGNAL(colorSelected(QColor)), colorTriangle, SLOT(setColor(QColor)));
    connect(colorTriangle, SIGNAL(colorChanged(QColor)), gradientEditor, SLOT(setColor(QColor)));
    connect(gradientEditor, SIGNAL(gradientChanged(QGradient)), this, SLOT(onGradientWidgetChanged(QGradient)));

    /* ------ Gradient Editor end ------ */

    layout->addItem(new QSpacerItem(10,10, QSizePolicy::Minimum, QSizePolicy::Expanding));
    applyButton = new QPushButton();
    layout->addWidget(applyButton);
    applyButton->setText("Apply");
    connect(applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
}

void VtkViewPropertyWidget::addSpacer(QString text)
{
    QFrame* frame = new QFrame();
    layout()->addWidget(frame);
    frame->setStyleSheet("padding-bottom: 2px; border-bottom: 1px solid #dddddd");

    QHBoxLayout* frameLayout = new QHBoxLayout(frame);
    frameLayout->setContentsMargins(0, 0, 0, 0);

    QLabel* label = new QLabel();
    label->setFont(spacerFont);
    label->setText("<font color='#555555'>" + text + "</font>");
    frameLayout->addWidget(label);
}

QComboBox* VtkViewPropertyWidget::createMappinfFunctionComboBox()
{
    QComboBox* cb = new QComboBox();
    cb->addItem(mappingFunctions["real"].Name, QVariant("real"));
    cb->addItem(mappingFunctions["imag"].Name, QVariant("imag"));
    cb->addItem(mappingFunctions["abs"].Name, QVariant("abs"));
    cb->addItem(mappingFunctions["arg"].Name, QVariant("arg"));
    return cb;
}

double VtkViewPropertyWidget::getWarpingScaling()
{
    return warpingScaling->value();
}

int VtkViewPropertyWidget::getFrequencyMinPercentage()
{
    return frequency->lowerValue();
}

int VtkViewPropertyWidget::getFrequencyMaxPercentage()
{
    return frequency->upperValue();
}

QGradient VtkViewPropertyWidget::getGradient()
{
    return gradientEditor->gradient();
}

VtkViewPropertyWidget::MappingFunction VtkViewPropertyWidget::getColorMapping()
{
    return mappingFunctions[gradientCompboBox->itemData(gradientCompboBox->currentIndex()).toString()];
}

VtkViewPropertyWidget::MappingFunction VtkViewPropertyWidget::getWarpingMapping()
{
    return mappingFunctions[warpingCompboBox->itemData(warpingCompboBox->currentIndex()).toString()];
}

float VtkViewPropertyWidget::getGradientMinValue()
{
    return gradientMinValue->text().toFloat();
}

float VtkViewPropertyWidget::getGradientMaxValue()
{
    return gradientMaxValue->text().toFloat();
}

void VtkViewPropertyWidget::onGradientWidgetChanged(QGradient g)
{
    gradientChangedFlag = true;
    // we emit this when the user clickes apply
    //emit gradientChanged(g, gradientMinValue->text().toFloat(), gradientMaxValue->text().toFloat());
}

void VtkViewPropertyWidget::onGradientMinMaxValueChanged(QString text)
{
    gradientChangedFlag = true;
    // we emit this when the user clickes apply
    //emit gradientChanged(gradientEditor->gradient(), gradientMinValue->text().toFloat(), gradientMaxValue->text().toFloat());
}

void VtkViewPropertyWidget::onGradientAutoMinMaxChanged(int state)
{
    if(state == Qt::Checked)
    {
        gradientMinValue->setEnabled(true);
        gradientMaxValue->setEnabled(true);
    }
    else if(state == Qt::Unchecked)
    {
        gradientMinValue->setDisabled(true);
        gradientMaxValue->setDisabled(true);
    }
}

void VtkViewPropertyWidget::setGradientMinMax(double min, double max)
{
    if(!autoMinMaxValue->isChecked())
    {
        double m = qMax(qAbs(min), qAbs(max));
        min = -m;
        max = m;
        gradientMinValue->setText(QString::number(min));
        gradientMaxValue->setText(QString::number(max));
        emit gradientChanged(gradientEditor->gradient(),
                             getGradientMinValue(),
                             getGradientMaxValue());
    }
}

void VtkViewPropertyWidget::setSpectrum(Spectrum s)
{
    spectrumMinFrequency = s.getEntries().first().Frequency;
    spectrumMaxFrequency = s.getEntries().last().Frequency;
    updateFrequencyMinMax(frequency->lowerValue(), frequency->upperValue());
}

void VtkViewPropertyWidget::updateFrequencyMinMax(int lower, int upper)
{
    double step = (spectrumMaxFrequency - spectrumMinFrequency) / 100.0;
    frequencyMin = spectrumMinFrequency + step * double(lower);
    frequencyMax = spectrumMinFrequency + step * double(upper);

    frequencyMinLabel->setText("<font color=\"#888888\" >" + QString::number(frequencyMin) + " 1/m<\font>");
    frequencyMaxLabel->setText("<font color=\"#888888\" >" + QString::number(frequencyMax) + " 1/m<\font>");

    frequencyMinMaxChangedFlag = true;
}

void VtkViewPropertyWidget::onWarpingCompboBoxChanged(int i)
{
    warpingMappingChangedFlag = true;
    //qDebug() << warpingCompboBox->itemData(i);
}

void VtkViewPropertyWidget::onGradientCompboBoxChanged(int i)
{
    colorMappingChangedFlag = true;
    //qDebug() << warpingCompboBox->itemData(i);
}

void VtkViewPropertyWidget::onWarpingScalingChanged(double i)
{
    warpingScalingChangedFlag = true;
}

void VtkViewPropertyWidget::onApplyClicked()
{
    if(gradientChangedFlag)
    {
        emit gradientChanged(gradientEditor->gradient(),
                             getGradientMinValue(),
                             getGradientMaxValue());
        gradientChangedFlag = false;
    }
    if(warpingMappingChangedFlag)
    {
        emit warpingMappingChanged(getWarpingMapping());
        warpingMappingChangedFlag = false;
    }
    if(colorMappingChangedFlag)
    {
        emit gradientMappingChanged(getColorMapping());
        colorMappingChangedFlag = false;
    }
    if(warpingScalingChangedFlag)
    {
        emit warpingScalingChanged(getWarpingScaling());
        warpingScalingChangedFlag = false;
    }
    if(frequencyMinMaxChangedFlag)
    {
        emit frequencyPercentageRangeChanged(frequency->lowerValue(), frequency->upperValue());
        frequencyMinMaxChangedFlag = false;
    }
    else // emit frequencyPercentageRangeChanged() triggers an updated
    {
        emit applyProperties();
    }
}
