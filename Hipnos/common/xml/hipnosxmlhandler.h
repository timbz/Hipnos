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

#ifndef HIPNOSQXMLHANDLER_H
#define HIPNOSQXMLHANDLER_H

#include <QXmlDefaultHandler>
#include "common/pipelinecomponent.h"

/**
 * @brief Base class for application specific QXmlDefaultHandler
 *
 */
class HipnosXmlHandler : public QXmlDefaultHandler
{

public:
    HipnosXmlHandler();

    bool fatalError(const QXmlParseException & exception);
    QString errorString() const;

protected:
    void setProperyForComponent(PipelineComponent* currentComponent, const QXmlAttributes &attributes, bool &success);
    PipelineComponent* getComponentFromAttributes(const QXmlAttributes &attributes);

    QString error;  
};

#endif // HIPNOSQXMLHANDLER_H
