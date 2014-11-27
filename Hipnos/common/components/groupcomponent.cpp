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

#include "groupcomponent.h"
#include "common/pipelinecomponentmanager.h"
#include "common/pipeline.h"

#include <QApplication>
#include <QFont>
#include <QPixmap>
#include <QPainter>

GroupComponent* GroupComponent::create(QString n, QList<PipelineComponent *> cpn, QColor c)
{
    // validate size > 0
    if(cpn.size() <= 0)
    {
        qDebug() << "Cannot create a component group with 0 elemets";
        return 0;
    }
    // validate linear stripe
    for(int i = 1; i < cpn.size(); i++)
    {
        if(cpn.at(i-1)->getNumberOfOutputConnections() != 1
                || cpn.at(i)->getNumberOfInputConnections() != 1)
        {
            qDebug() << "Cannot create a component group from a non linear stripe";
            return 0;
        }
        Pipeline::pipelineConnection(cpn.at(i-1), cpn.at(i));
    }
    return new GroupComponent(n, cpn, c);
}

GroupComponent::GroupComponent(QString n, QList<PipelineComponent*> cpn, QColor c) :
    PipelineComponent()
{
    typeName = n;
    components = cpn;
    color = c;
    setName(n);
    updateLength();
    setNumberOfInputConnections(components.first()->getNumberOfInputConnections());
    setNumberOfOutputConnections(components.last()->getNumberOfOutputConnections());

    if(getNumberOfInputConnections() > 0)
    {
        // we connect the proxy component to the input of the first component with a non cached connection
        inputProxy = new InnerInputProxyComponent(this);
        for(int i = 0; i < getNumberOfInputConnections(); i++)
        {
            PipelineConnection* con = new PipelineConnection(inputProxy, components.first(), true);
            inputProxy->setOutputConnection(con, i);
            components.first()->setInputConnection(con, i);
        }
    }
    else
    {
        inputProxy = 0;
    }

    // we add a sink to the last component to fetch the data later on
    for(int i = 0; i < components.last()->getNumberOfOutputConnections(); i++)
    {
        components.last()->setOutputConnection(new PipelineSink(components.last()), i);
    }

    // we take the ownership of the PipelineConnections
    PipelineComponent* comp = inputProxy;
    while(comp)
    {
        PipelineComponent* next = comp->getOutputConnection()->getOutput();
        for(int i = 0; i < comp->getNumberOfInputConnections(); i++)
        {
            connections.insert(comp->getInputConnection(i));
        }
        comp = next;
    }
}

GroupComponent::~GroupComponent()
{
    foreach(PipelineConnection* con, connections)
    {
        delete con;
    }
    if(inputProxy)
        delete inputProxy;
}

void GroupComponent::updateLength()
{
    length = 0;
    foreach(PipelineComponent* p, components)
    {
        length += p->getLength();
    }
}

QList<QString> GroupComponent::getDependencyKeys()
{
    QList<QString> dep;
    foreach(PipelineComponent* p, components)
    {
        if(!p->isBasicComponent()) // ist a grouped component
            dep << p->getType();
    }
    return dep;
}

QString GroupComponent::getType()
{
    return typeName;
}

QColor GroupComponent::getColor()
{
    return color;
}

QIcon GroupComponent::getIcon()
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
    painter.setBrush(QBrush(color));
    painter.drawRoundedRect(0, 0, iconSize-1 ,iconSize-1, 10, 10, Qt::RelativeSize);
    painter.drawText(x, y, getType());
    return QIcon(pixmap);
}

PipelineComponent* GroupComponent::clone()
{
    QList<PipelineComponent*> clones;
    // clone inner components
    foreach(PipelineComponent* c, components)
    {
        clones << c->clone();
    }
    // connect inner components
    for(int i = 1; i < clones.size(); i++)
    {
        Pipeline::pipelineConnection(clones.at(i-1), clones.at(i));
    }
    // create clone
    GroupComponent* c = new GroupComponent(getType(), clones, color);

    return c;
}

void GroupComponent::setProperty(Property p)
{
    QStringList split = p.getName().split("::");
    if(split.size() > 1)
    {
        QString subComponentName = split.at(0);
        p.setName(p.getName().remove(subComponentName + "::"));
        foreach(PipelineComponent* c, components)
        {
            if(subComponentName == c->getName())
            {
                c->enqueueProperty(p);
                c->setChanged();
            }
        }
    }
    updateLength();
}

QList<PipelineComponent::Property> GroupComponent::getProperties()
{
    QList<Property> l;
    foreach(PipelineComponent* c, components)
    {
        foreach(PipelineComponent::Property p, c->getProperties())
        {
            p.setName(c->getName() + "::" + p.getName());
            p.setComponent(this);
            l << p;
        }
    }
    return l;
}

QList<PipelineComponent*> GroupComponent::ungroup()
{
    QList<PipelineComponent*> tmp = components;
    components.clear();
    return tmp;
}

bool GroupComponent::isBasicComponent()
{
    return false;
}

void GroupComponent::flush()
{
    components.first()->flush();
    PipelineComponent::flush();
}

void GroupComponent::propagation(double z, PipelineDataType dataType)
{
    if(z >= length)
    {
        for(int i = 0; i < components.last()->getNumberOfOutputConnections(); i++)
        {
            getOutputConnection(i)->setData(
                        components.last()->getOutputConnection(i)->getData(dataType));
        }
    }
    else
    {
        PipelineComponent* c = components.first();
        while(c->getOutputConnection()->getOutput()
              && z >= c->getLength())
        {
            z -= c->getLength();
            c = c->getOutputConnection()->getOutput();
        }
        // only linear component chain
        getOutputConnection()->setData(
                    c->getOutputConnection()->fetchIntermediatDataFromInput(dataType, z));
    }
}

void GroupComponent::gaussPropagation(double z)
{
    propagation(z, DT_GAUSS);
}

void GroupComponent::fourierPropagation(double z)
{
    propagation(z, DT_FOURIER);
}

QList<PipelineComponent*> GroupComponent::getComponents()
{
    return components;
}

void GroupComponent::setChanged()
{
    components.front()->setChanged();
    PipelineComponent::setChanged();
}
