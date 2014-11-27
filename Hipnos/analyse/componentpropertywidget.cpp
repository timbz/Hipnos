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

#include "componentpropertywidget.h"
#include "common/widgets/infobutton.h"
#include "common/widgets/spinbox.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QDebug>
#include <QLineEdit>
#include <QPushButton>
#include <QGridLayout>
#include <QComboBox>

ComponentPropertyWidget::ComponentPropertyWidget(QWidget *parent) :
    QWidget(parent)
{
    xScale = 1;
    QVBoxLayout* lt = new QVBoxLayout(this);
    lt->setSpacing(9);
    lt->setContentsMargins(9, 9, 9, 6);
    reset();
}

ComponentPropertyWidget::~ComponentPropertyWidget()
{
    foreach(PropertyChangedHandler* h, handlers)
    {
        delete h;
    }
}

void ComponentPropertyWidget::clear()
{
    //clear
    while(!handlers.isEmpty())
    {
        delete handlers.front();
        handlers.pop_front();
    }
    QLayoutItem *child;
    while ((child = layout()->takeAt(0)) != 0) {
        delete child->widget();
        delete child;
    }
}

void ComponentPropertyWidget::setXScale(qreal s)
{
    xScale = s;
}

void ComponentPropertyWidget::setChart(ChartSceneItem *chart)
{
    clear();
    if(chart)
    {
        QLabel* title = new QLabel("<b>Chart</b>");
        layout()->addWidget(title);

        QDoubleSpinBox* ysize = new QDoubleSpinBox();
        layout()->addWidget(ysize);
        ysize->setRange(5, 500);
        ysize->setDecimals(0);
        ysize->setSingleStep(5);
        ysize->setValue(chart->getYAxisSize());
        ysize->setPrefix("Y Axis size: ");
        ysize->setSuffix(" px");
        connect(ysize, SIGNAL(valueChanged(double)), chart, SLOT(setYAxisSize(double)));

        QDoubleSpinBox* stepSize = new QDoubleSpinBox();
        layout()->addWidget(stepSize);
        stepSize->setRange(1, 1000);
        stepSize->setDecimals(1);
        stepSize->setSingleStep(5);
        stepSize->setValue(chart->getStepSize());
        stepSize->setPrefix("Sampling step: ");
        stepSize->setSuffix(" m");
        connect(stepSize, SIGNAL(valueChanged(double)), chart, SLOT(setStepSize(double)));

        layout()->addItem(new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding));
    }
}

void ComponentPropertyWidget::reset()
{
    clear();

    QLabel* title = new QLabel("<b>Beamline</b>");
    layout()->addWidget(title);

    QDoubleSpinBox* xscaling = new QDoubleSpinBox();
    layout()->addWidget(xscaling);
    xscaling->setRange(0.1, 100);
    xscaling->setDecimals(1);
    xscaling->setSingleStep(0.1);
    xscaling->setValue(xScale);
    xscaling->setPrefix("Z Scaling: ");
    connect(xscaling, SIGNAL(valueChanged(double)), this, SLOT(onXScaleChanged(double)));

    layout()->addItem(new QSpacerItem(10, 20));
    QLabel* txt = new QLabel("Select a component to display properties");
    txt->setWordWrap(true);
    layout()->addWidget(txt);

    layout()->addItem(new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding));
}

void ComponentPropertyWidget::setComponent(PipelineComponent* c)
{
    component = c;
    clear();
    if(component)
    {
        QLabel* componentType = new QLabel("<b>Type: " + component->getType() + "</b>");
        layout()->addWidget(componentType);

        QFrame* stringEdit = new QFrame();
        QHBoxLayout* l = new QHBoxLayout(stringEdit);
        l->setContentsMargins(0, 9, 0, 9);
        QLabel* label = new QLabel("Name");
        l->addWidget(label);
        QLineEdit* nameEdit = new QLineEdit(component->getName());
        nameEdit->setValidator(&nameValidator);
        connect(nameEdit, SIGNAL(textEdited(QString)), this, SLOT(onNameChanged(QString)));
        l->addWidget(nameEdit);
        layout()->addWidget(stringEdit);

        QFrame* propsFrame = new QFrame();
        QGridLayout* propsLayout = new QGridLayout(propsFrame);
        propsLayout->setAlignment(Qt::AlignTop);
        propsLayout->setContentsMargins(0, 0, 0, 0);
        layout()->addWidget(propsFrame);

        int row = 0;
        foreach(PipelineComponent::Property p, component->getProperties())
        {
            InfoButton* info = new InfoButton(p.getName(), p.getDescription());
            propsLayout->addWidget(info, row, 0);
            if(p.getType() == PipelineComponent::Property::PT_STRING)
            {
                QHBoxLayout* sl = new QHBoxLayout();
                sl->setContentsMargins(0, 9, 0, 9);
                QLabel* label = new QLabel(p.getName());
                sl->addWidget(label);
                QLineEdit* edit = new QLineEdit(p.getStringValue());
                sl->addWidget(edit);
                propsLayout->addLayout(sl, row, 1);
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, edit);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            else if(p.getType() == PipelineComponent::Property::PT_INT)
            {
                SpinBox* spin = new SpinBox();
                propsLayout->addWidget(spin, row, 1);
                spin->setRange(p.getIntMinValue(), p.getIntMaxValue());
                spin->setValue(p.getIntValue());
                spin->setPrefix(p.getName() + ": ");
                spin->setValidator(p.getValidator());
                spin->setSingleStep(p.getSingleStep().toInt());
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, spin);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            else if(p.getType() == PipelineComponent::Property::PT_DOUBLE)
            {
                QDoubleSpinBox* spin = new QDoubleSpinBox();
                propsLayout->addWidget(spin, row, 1);
                spin->setDecimals(6);
                spin->setRange(p.getDoubleMinValue(), p.getDoubleMaxValue());
                spin->setValue(p.getDoubleValue());
                spin->setPrefix(p.getName() + ": ");
                //spin->setValidator(p.getValidator());
                spin->setSuffix(" " + p.getUnit());
                spin->setSingleStep(p.getSingleStep().toDouble());
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, spin);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            else if(p.getType() == PipelineComponent::Property::PT_LIST)
            {
                QComboBox* cb = new QComboBox();
                propsLayout->addWidget(cb, row, 1);
                int i = 0;
                foreach(QString o, p.getPropertyOptions())
                {
                    cb->insertItem(i++, o);
                }
                int index = cb->findText(p.getSelectedValue());
                if(index >= 0)
                    cb->setCurrentIndex(index);
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, cb);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            else if(p.getType() == PipelineComponent::Property::PT_FILE)
            {
                QHBoxLayout* fileEditLayout = new QHBoxLayout();
                propsLayout->addLayout(fileEditLayout, row, 1);
                QLabel* fileLable = new QLabel(p.getName());
                fileEditLayout->addWidget(fileLable);
                QLineEdit* fileEdit = new QLineEdit(p.getFileName());
                fileEditLayout->addWidget(fileEdit);
                QPushButton* browse = new QPushButton("...");
                browse->setMaximumWidth(40);
                fileEditLayout->addWidget(browse);
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, fileEdit, browse);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            else if(p.getType() == PipelineComponent::Property::PT_SPECTRUM)
            {
                QHBoxLayout* spectrumEditLayout = new QHBoxLayout();
                propsLayout->addLayout(spectrumEditLayout, row, 1);
                QLabel* l = new QLabel(p.getName());
                spectrumEditLayout->addWidget(l);
                QPushButton* b = new QPushButton("Edit");
                spectrumEditLayout->addWidget(b);
                PropertyChangedHandler* handler = new PropertyChangedHandler(p, b);
                connect(handler, SIGNAL(propertyChanged(PipelineComponent::Property)), this, SLOT(onPropertyChanged(PipelineComponent::Property)));
                handlers << handler;
            }
            row++;
        }
        propsLayout->addItem(new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding), row, 0);
    }
}

void ComponentPropertyWidget::onPropertyChanged(PipelineComponent::Property p)
{
    emit propertyChanged(p);
}

void ComponentPropertyWidget::onNameChanged(QString n)
{
    if(component)
        component->setName(n);
}

void ComponentPropertyWidget::onXScaleChanged(double s)
{
    xScale = s;
    emit xScaleChanged(s);;
}
