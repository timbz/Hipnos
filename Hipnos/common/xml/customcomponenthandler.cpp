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

#include "customcomponenthandler.h"
#include "common/pipelinecomponentmanager.h"

CustomComponentHandler::CustomComponentHandler()
{
}

bool CustomComponentHandler::startDocument()
{
    return true;
}


bool CustomComponentHandler::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &attributes)
{
    if(qName == "GroupComponent")
    {
        currentComponentList.clear();

        currentGroupName = attributes.value("name");
        if(currentGroupName.isEmpty())
        {
            error = "GroupComponent has no attribute \"name\"";
            return false;
        }
        QString colorString = attributes.value("color");
        if(colorString.isEmpty())
        {
            error = "GroupComponent has no attribute \"color\"";
            return false;
        }
        QStringList rgba = colorString.split(",");
        if(rgba.size() != 4)
        {
            error = "Could not parse color string " + colorString;
        }
        currentGroupColor = QColor(rgba.at(0).toInt(), rgba.at(1).toInt(), rgba.at(2).toInt(), rgba.at(3).toInt());

    }
    else if(qName == "Component")
    {
        currentComponent = getComponentFromAttributes(attributes);
        if(currentComponent == 0)
        {
            return false;
        }
    }
    else if(qName == "Property")
    {
        bool ok;
        setProperyForComponent(currentComponent, attributes, ok);
        if(!ok)
        {
            return false;
        }
    }
    return true;
}


bool CustomComponentHandler::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
    if(qName == "GroupComponent")
    {
        GroupComponent* group = GroupComponent::create(currentGroupName, currentComponentList, currentGroupColor);
        if(group == 0)
        {
            error = "Error creating GroupComponent";
            foreach(PipelineComponent* c, currentComponentList)
            {
                delete c;
            }
            return false;
        }
        PipelineComponentManager::getInstance().addCustomComponent(group);
    }
    else if(qName == "Component")
    {
        currentComponentList << currentComponent;
    }
    return true;
}
