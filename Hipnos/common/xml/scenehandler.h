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

#ifndef SCENEHANDLER_H
#define SCENEHANDLER_H

#include "hipnosxmlhandler.h"
#include "design/designerscene.h"

/**
 * @brief XML handler used to parse saved component pipelines
 *
 */
class SceneHandler : public HipnosXmlHandler
{

public:
    SceneHandler(DesignerScene* scene);

    bool startDocument();
    bool startElement(const QString &namespaceURI, const QString &localName,
                      const QString &qName, const QXmlAttributes &attributes);
    bool endElement(const QString &namespaceURI, const QString &localName,
                    const QString &qName);

private:
    DesignerScene* scene;  
    int slot;  
    PipelineComponent* currentComponent;  
};

#endif // SCENEHANDLER_H
