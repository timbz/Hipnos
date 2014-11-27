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

#include "pipelinecomponentmanager.h"
#include "pipelinecomponent.h"
#include "filesysteminterface.h"
#include "components/groupcomponent.h"
#include "components/aperturecomponent.h"
#include "components/propagationcomponent.h"
#include "components/gaussianbeamsourcecomponent.h"
#include "components/thinlenscomponent.h"
#include "components/csvsourcecomponent.h"
#include <QDebug>
#include <QMessageBox>

PipelineComponentManager::PipelineComponentManager()
{
    qDebug() << "Initializing PipelineComponentManager";
    PropagationComponent* drift = new PropagationComponent();
    basicComponentes[drift->getType()] = drift;
    GaussianBeamSourceComponent* source = new GaussianBeamSourceComponent();
    basicComponentes[source->getType()] = source;
    ThinLensComponent* thinLens = new ThinLensComponent();
    basicComponentes[thinLens->getType()] = thinLens;
    CircularApertureComponent* circularAperture = new CircularApertureComponent();
    basicComponentes[circularAperture->getType()] = circularAperture;
    CircularObstacleComponent* circularObstacle = new CircularObstacleComponent();
    basicComponentes[circularObstacle->getType()] = circularObstacle;
    RectangularApertureComponent* rectangularAperture = new RectangularApertureComponent();
    basicComponentes[rectangularAperture->getType()] = rectangularAperture;
    RectangularObstacleComponent* rectangularObstacle = new RectangularObstacleComponent();
    basicComponentes[rectangularObstacle->getType()] = rectangularObstacle;
    CsvSourceComponent* csvSource = new CsvSourceComponent();
    basicComponentes[csvSource->getType()] = csvSource;
}

PipelineComponentManager::~PipelineComponentManager()
{
    foreach(PipelineComponent* c, basicComponentes)
    {
        delete c;
    }
    foreach(GroupComponent* g, customComponents)
    {
        foreach(PipelineComponent* c, g->getComponents())
        {
            delete c;
        }
        delete g;
    }
}

QHash<QString, PipelineComponent*>& PipelineComponentManager::getBasicComponents()
{
    return basicComponentes;
}

QHash<QString, GroupComponent*>& PipelineComponentManager::getCustomComponents()
{
    return customComponents;
}

PipelineComponent* PipelineComponentManager::getComponentByTypeName(QString name)
{
    if(basicComponentes.contains(name))
        return basicComponentes[name];
    if(customComponents.contains(name))
        return customComponents[name];
    return 0;
}

void PipelineComponentManager::loadCustomComponents()
{
    FileSystemInterface::getInstance().loadCustomComponents();
}

void PipelineComponentManager::saveCustomComponents()
{
    FileSystemInterface::getInstance().saveCustomComponents(customComponents);
    qDebug() << "Saved " << customComponents.size() << " custom components";
}

void PipelineComponentManager::addCustomComponent(GroupComponent* g)
{
    customComponents[g->getType()] = g;
}

bool PipelineComponentManager::deleteCustomComponent(QString name)
{
    if(customComponents.contains(name))
    {
        foreach(GroupComponent* g, customComponents)
        {
            if(g->getDependencyKeys().contains(name))
            {
                QMessageBox::warning(0, "Error", "Could not delete " + name + ": " +
                                     g->getType() + " depends this component",
                                     QMessageBox::Close, QMessageBox::NoButton);
                return false;
            }
        }
        delete customComponents[name];
        customComponents.remove(name);
    }
    return true;
}

bool PipelineComponentManager::isTypeNameValid(QString name)
{
    if(name.isNull() || name.isEmpty() || customComponents.contains(name) || basicComponentes.contains(name))
        return false;
    return true;
}
