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

#include "hipnosxmlhandler.h"
#include "common/statusmessage.h"
#include "common/pipelinecomponentmanager.h"

HipnosXmlHandler::HipnosXmlHandler()
{
}

bool HipnosXmlHandler::fatalError (const QXmlParseException & exception)
{
    qWarning() << "Fatal error on line" << exception.lineNumber()
               << ", column" << exception.columnNumber() << ":"
               << exception.message();
    StatusMessage::show("Error while opening file: " + exception.message());
    return false;
}

QString HipnosXmlHandler::errorString() const
{
    return error;
}

PipelineComponent* HipnosXmlHandler::getComponentFromAttributes(const QXmlAttributes &attributes)
{
    QString type = attributes.value("type");
    if(type.isEmpty())
    {
        error = "Component has no attribute \"type\"";
        return 0;
    }
    QString name = attributes.value("name");
    if(name.isEmpty())
    {
        error = "Component has no attribute \"name\"";
        return 0;
    }
    PipelineComponent* prototype = PipelineComponentManager::getInstance().getComponentByTypeName(type);
    if(prototype)
    {
        PipelineComponent* c = prototype->clone();
        c->setName(name);
        return c;
    }
    else
    {
        error = "Could not find component " + type;
        return 0;
    }
}

void HipnosXmlHandler::setProperyForComponent(PipelineComponent* currentComponent, const QXmlAttributes &attributes, bool &success)
{
    success = true;
    QString name = attributes.value("name");
    if(name.isEmpty())
    {
        error = "Property has no attribute \"name\"";
        success = false;
        return;
    }
    QString value = attributes.value("value");
    if(value.isEmpty())
    {
        error = "Property has no attribute \"value\"";
        success = false;
        return;
    }
    foreach(PipelineComponent::Property p, currentComponent->getProperties())
    {
        if(p.getName() == name)
        {
            success = p.currentValueFromString(value);
            if(success)
            {
                currentComponent->setProperty(p);
            }
            return;
        }
    }
}


