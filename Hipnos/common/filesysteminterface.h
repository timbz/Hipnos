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

#ifndef FILESYSTEMINTERFACE_H
#define FILESYSTEMINTERFACE_H

#include <QDir>
#include "pipelinecomponent.h"
#include "components/groupcomponent.h"
#include "design/designerscene.h"

/**
 * @brief Singelton class that manages the interaction of the application
 *          with the file system
 */
class FileSystemInterface
{

public:
    /**
     * @brief Returns a reference to the singelton instance
     *
     * @return FileSystemInterface & A reference to the instance
     */
    static FileSystemInterface& getInstance()
    {
        static FileSystemInterface instance;
        return instance;
    }

    QString getLoggingPath();
    void loadCustomComponents();
    void saveCustomComponents(QHash<QString, GroupComponent*> c);
    void loadDesignerScene(DesignerScene* scene);
    void saveDesignerScene(DesignerScene* scene);
    QDir getHipnosHome();
    QList<MathPlugin*> loadMathPlugins();

private:
    FileSystemInterface();

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    FileSystemInterface(FileSystemInterface const&);

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    void operator=(FileSystemInterface const&);

    QString componentToXML(PipelineComponent* c, uint getindent = 0);    QString getIndent(uint indent);

    QDir hipnosHome;  
    QString logFilePath;  
    QString customComponentsFilePath;  
    QString hipnosFileExtension;  
    QDir pluginDir;  
};

#endif // FILESYSTEMINTERFACE_H
