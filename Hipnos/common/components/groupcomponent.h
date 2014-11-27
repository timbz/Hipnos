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

#ifndef GROUPPIPELINECOMPONENT_H
#define GROUPPIPELINECOMPONENT_H

#include "common/pipelinecomponent.h"

/**
 * @brief A PipelineComponent that groups together a list of other PipelineComponent
 *
 */
class GroupComponent : public PipelineComponent
{

public:
    static GroupComponent* create(QString n, QList<PipelineComponent*> cpn, QColor c);
    ~GroupComponent();

    QString getType();
    PipelineComponent* clone();
    QIcon getIcon();
    void setProperty(Property p);
    QList<Property> getProperties();
    void gaussPropagation(double z);
    void fourierPropagation(double z);
    bool isBasicComponent();
    void flush();
    void setChanged();
    void propagation(double z, PipelineDataType dataType);
    QList<PipelineComponent*> ungroup();
    QList<PipelineComponent*> getComponents();
    QColor getColor();
    void updateLength();
    QList<QString> getDependencyKeys();

private:

    /**
     * @brief A PipelineComponent only used by a GroupComponent to forward the imput data
     *      to the first inner component
     */
    class InnerInputProxyComponent : public PipelineComponent
    {
    private:
        GroupComponent* subject;  

    public:
        /**
         * @brief Constructs a new InnerInputProxyComponent
         *
         * @param s The subject
         */
        InnerInputProxyComponent(GroupComponent* s) :
            PipelineComponent()
        {
            subject = s;
            setNumberOfOutputConnections(subject->getNumberOfInputConnections());
        }

        /**
         * @brief Fetches data from the imput connection of the subject an forwars it to his
         *      output connection
         *
         * @param z
         */
        virtual void gaussPropagation(double z)
        {
            for(int i = 0; i < getNumberOfOutputConnections(); i++)
            {
                getOutputConnection(i)->setData(
                            subject->getInputConnection(i)->getData(DT_GAUSS));
            }
        }


        /**
         * @brief Fetches data from the imput connection of the subject an forwars it to his
         *      output connection
         *
         * @param z
         */
        virtual void fourierPropagation(double z)
        {
            for(int i = 0; i < getNumberOfOutputConnections(); i++)
            {
                getOutputConnection(i)->setData(
                            subject->getInputConnection(i)->getData(DT_FOURIER));
            }
        }

        /**
         * @brief Unsupported operation (this component is only used internally by the GroupComponent class
         *
         * @return QString
         */
        virtual QString getType()
        {
            qFatal("unsupported operation");
            return QString();
        }

        /**
         * @brief Unsupported operation (this component is only used internally by the GroupComponent class
         *
         * @return PipelineComponent *
         */
        virtual PipelineComponent* clone()
        {
            qFatal("unsupported operation");
            return 0;
        }

        /**
         * @brief Unsupported operation (this component is only used internally by the GroupComponent class
         *
         * @param p
         */
        virtual void setProperty(Property p)
        {
            qFatal("unsupported operation");
        }

        /**
         * @brief Unsupported operation (this component is only used internally by the GroupComponent class
         *
         * @return QList<Property>
         */
        virtual QList<Property> getProperties()
        {
            qFatal("unsupported operation");
            return QList<Property>();
        }
    };

/**
 * @brief
 *
 * @param n
 * @param cpn
 * @param c
 */
/**
 * @brief
 *
 * @param n
 * @param cpn
 * @param c
 */
    GroupComponent(QString n, QList<PipelineComponent*> cpn, QColor c);

    InnerInputProxyComponent* inputProxy;  
    QList<PipelineComponent*> components;  
    QList<PipelineConnection*> innerInputConnections;  
    QColor color;  
    QString typeName;  
    QSet<PipelineConnection*> connections;  
};

#endif // GROUPPIPELINECOMPONENT_H
