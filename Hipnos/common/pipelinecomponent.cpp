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

#include "pipelinecomponent.h"

#include <QApplication>
#include <QFont>
#include <QPixmap>
#include <QPainter>

QSet<QString> PipelineComponent::uniqueComponentNames;

PipelineComponent::PipelineComponent()
{
    length = 0;
    changed = true;
    name = "";
}

PipelineComponent::~PipelineComponent()
{
    freeUniqueName(name);
}

PipelineConnection* PipelineComponent::getInputConnection(int i)
{
    Q_ASSERT_X(i >= 0 && i < input.size(), "PipelineComponent::getInputComponent", "index out of range");
    return input[i];
}

PipelineConnection* PipelineComponent::getOutputConnection(int i)
{
    Q_ASSERT_X(i >= 0 && i < output.size(), "PipelineComponent::getOutputComponent", "index out of range");
    return output[i];
}

void PipelineComponent::setInputConnection(PipelineConnection *c, int i)
{
    Q_ASSERT_X(i >= 0 && i < input.size(), "PipelineComponent::setInputComponent", "index out of range");
    input[i] = c;
}

void PipelineComponent::setOutputConnection(PipelineConnection *c, int i)
{
    Q_ASSERT_X(i >= 0 && i < output.size(), "PipelineComponent::setOutputComponent", "index out of range");
    output[i] = c;
}

bool PipelineComponent::isBasicComponent()
{
    return true;
}

QString PipelineComponent::getLabelText()
{
    return getType() + " [In: " + QString::number(input.size()) + ", Out: " + QString::number(output.size()) + "]";
}

int PipelineComponent::getNumberOfInputConnections()
{
    return input.size();
}

int PipelineComponent::getNumberOfOutputConnections()
{
    return output.size();
}

void PipelineComponent::flush()
{
    for(int i = 0; i < getNumberOfOutputConnections(); i++)
    {
        PipelineConnection* c = getOutputConnection(i);
        if(c)
           c->flush();
    }
}

void PipelineComponent::setNumberOfInputConnections(int n)
{
    Q_ASSERT_X(n <= 1, "PipelineComponent::setNumberOfInputConnections", "number of input connections is currently limited to 1");
    input.resize(n);
    for(int i = 0; i < n; i++)
        input[i] = 0;
}

void PipelineComponent::setNumberOfOutputConnections(int n)
{
    Q_ASSERT_X(n <= 1, "PipelineComponent::setNumberOfOutputConnections", "number of output connections is currently limited to 1");
    output.resize(n);
    for(int i = 0; i < n; i++)
        output[i] = 0;
}

double PipelineComponent::getLength()
{
    return length;
}

QString PipelineComponent::getName()
{
    if(name == "")
        setName(getUniqueComponentName(getType()));
    return name;
}

void PipelineComponent::setName(QString n)
{
    if(isNameValid(n))
    {
        freeUniqueName(name);
        name = n;
        uniqueComponentNames.insert(n);
    }
    else
    {
        setName(getUniqueComponentName(n));
    }
}

QIcon PipelineComponent::getIcon()
{
    int padding = 5; // 3 pixel padding
    QFont font = QApplication::font();
    font.setPixelSize(30);
    QFontMetrics fm(font);
    QSize textSize = fm.size(Qt::TextSingleLine, getType());
    int iconSize, x, y;
    if(textSize.width() > textSize.height())
    {
        iconSize = textSize.width()+2*padding;
        x = padding;
        y = iconSize/2 + textSize.height()/4;
    }
    else
    {
        iconSize = textSize.height()+2*padding;
        y = iconSize/2 + 2*padding;
        x = iconSize/2 - textSize.width()/2;
    }
    QPixmap pixmap(iconSize, iconSize);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setFont(font);
    painter.setBrush(QBrush(Qt::white));
    painter.drawRoundedRect(0, 0, iconSize-1 ,iconSize-1, 10, 10, Qt::RelativeSize);
    painter.drawText(x, y, getType());
    return QIcon(pixmap);
}

bool PipelineComponent::isChanged()
{
    return changed;
}

void PipelineComponent::setChanged()
{
    changed = true;
    for(int i  = 0; i < output.size(); i++)
    {
        if(output[i])
        {
            output[i]->flush();
        }
    }
}

void PipelineComponent::computePropagation(PipelineDataType dataType, double z)
{
    while(!queuedProperties.isEmpty())
    {
        setProperty(queuedProperties.dequeue());
    }
    switch(dataType)
    {
    case DT_GAUSS:
        gaussPropagation(z);
        break;
    case DT_FOURIER:
        fourierPropagation(z);
        break;
    }
    changed = false;
}

void PipelineComponent::enqueueProperty(Property p)
{
    queuedProperties.enqueue(p);
}

bool PipelineComponent::isNameValid(QString n)
{
    return !uniqueComponentNames.contains(n);
}

QString PipelineComponent::getUniqueComponentName(QString type)
{
    uint i = 0;
    QString uniqueName = type + "_" + QString::number(i);
    while(uniqueComponentNames.contains(uniqueName))
    {
        i++;
        uniqueName = type + "_" + QString::number(i);
    }
    return uniqueName;
}

void PipelineComponent::freeUniqueName(QString n)
{
    uniqueComponentNames.remove(n);
}
