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

#ifndef PIPELINECOMPONENTMANAGER_H
#define PIPELINECOMPONENTMANAGER_H

#include <QString>
#include <QHash>
#include <QSet>

class PipelineComponent;
class GroupComponent;

/**
 * @brief Singelton class that manages all available PipelineComponents
 *
 */
class PipelineComponentManager
{

public:
    /**
     * @brief Returns a reference to the current instance
     *
     * @return PipelineComponentManager &
     */
    static PipelineComponentManager& getInstance()
    {
        static PipelineComponentManager instance;
        return instance;
    }
    ~PipelineComponentManager();

    QHash<QString, PipelineComponent*>& getBasicComponents();
    QHash<QString, GroupComponent*>& getCustomComponents();
    PipelineComponent* getComponentByTypeName(QString name);
    void loadCustomComponents();
    void saveCustomComponents();
    bool deleteCustomComponent(QString name);
    void addCustomComponent(GroupComponent* g);
    bool isTypeNameValid(QString name);

private:
    PipelineComponentManager();

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    PipelineComponentManager(PipelineComponentManager const&);

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    void operator=(PipelineComponentManager const&);

    QHash<QString, PipelineComponent*> basicComponentes;  
    QHash<QString, GroupComponent*> customComponents;  

};
#endif // PIPELINECOMPONENTMANAGER_H
