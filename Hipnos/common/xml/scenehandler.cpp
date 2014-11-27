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

#include "scenehandler.h"

SceneHandler::SceneHandler(DesignerScene* s)
{
    scene = s;
}

bool SceneHandler::startDocument()
{
    slot = 0;
    return true;
}


bool SceneHandler::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &attributes)
{
    if(qName == "Stripe")
    {
        QString n = attributes.value("slots");
        if(n.isEmpty())
        {
            error = "Stripe has no attribute \"slots\"";
            return false;
        }
        bool ok;
        int slotsNumber = n.toInt(&ok);
        if(!ok)
        {
            error = "Could not parse slot number from " + n;
        }
        scene->getStripe()->getComponetSceneItems().resize(slotsNumber);
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


bool SceneHandler::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
    if(qName == "Slot")
    {
        slot++;
    }
    else if(qName == "Component")
    {
        scene->getStripe()->addComponentToSlot(currentComponent, slot);
    }
    return true;
}
