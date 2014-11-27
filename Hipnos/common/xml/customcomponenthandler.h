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

#ifndef CUSTOMCOMPONENTHANDLER_H
#define CUSTOMCOMPONENTHANDLER_H

#include "hipnosxmlhandler.h"
#include "common/components/groupcomponent.h"

/**
 * @brief XML handler used to parse saved custom componets
 *
 */
class CustomComponentHandler : public HipnosXmlHandler
{

public:
    CustomComponentHandler();

    bool startDocument();
    bool startElement(const QString &namespaceURI, const QString &localName,
                      const QString &qName, const QXmlAttributes &attributes);

    bool endElement(const QString &namespaceURI, const QString &localName,
                    const QString &qName);

private:
    QString currentGroupName;  
    QColor currentGroupColor;  
    PipelineComponent* currentComponent;  
    QList<PipelineComponent*> currentComponentList;  
};

#endif // CUSTOMCOMPONENTHANDLER_H
