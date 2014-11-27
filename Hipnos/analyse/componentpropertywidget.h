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

#ifndef COMPONENTPROPERTYWIDGET_H
#define COMPONENTPROPERTYWIDGET_H

#include <QWidget>
#include "common/pipelinecomponent.h"
#include "propertychangedhandler.h"
#include "common/statusmessage.h"
#include "chartsceneitem.h"

/**
 * @brief QWidget that displays controls to edit the properties of a PipelineComponet
 *
 */
class ComponentPropertyWidget : public QWidget
{
    Q_OBJECT
public:

    /**
     * @brief This QValidator implementations checks if a given PipelienComponent name is already in use or not.
     *                For consistency we have to avoid duplicates.
     */
    class NameValidator : public QValidator
    {

         /**
          * @brief Validates a specific name
          *
          * @param input The name to validate
          * @param pos  unused in this implementation
          * @return State returns QValidator::Acceptable if the name is free, otherwise it returns QValidator::Invalid
          */
         State validate(QString &input, int &pos ) const
         {
             if(PipelineComponent::isNameValid(input))
             {
                 return Acceptable;
             }
             else
             {
                 StatusMessage::show("Invalid name");
                 return Invalid;
             }
         }
    };

    explicit ComponentPropertyWidget(QWidget *parent = 0);
    ~ComponentPropertyWidget();

    void setComponent(PipelineComponent* c);
    void setChart(ChartSceneItem* chart);
    void reset();
    void clear();

signals:
    void propertyChanged(PipelineComponent::Property p);
    void xScaleChanged(double s);

public slots:
    void setXScale(qreal s);
    void onPropertyChanged(PipelineComponent::Property p);
    void onNameChanged(QString n);
    void onXScaleChanged(double s);

private:
    qreal xScale;  
    PipelineComponent* component;  
    NameValidator nameValidator;  
    QList<PropertyChangedHandler*> handlers;  
};

#endif // COMPONENTPROPERTYWIDGET_H
