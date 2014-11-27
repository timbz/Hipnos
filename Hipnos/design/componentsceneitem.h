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

#ifndef DESIGNERSCENEITEM_H
#define DESIGNERSCENEITEM_H

#include <QGraphicsObject>
#include <QPainter>
#include <QGraphicsSceneContextMenuEvent>
#include "common/pipelinecomponent.h"

class ComponentStripeSceneItem;

/**
 * @brief Implementaion of a QGraphicsObject that draws a PipelineComponent to the DesignerScene
 *
 */
class ComponentSceneItem : public QGraphicsObject
{
    Q_OBJECT
public:


    /**
     * @brief Operator used to compare and sort a list of ComponentSceneItem
     *
     */
    struct LessThan
    {
        bool operator()(const ComponentSceneItem* a, const ComponentSceneItem* b) const
        {
            if(a->x() < b->x()) return true;
            return false;
        }
    };

    ComponentSceneItem(ComponentStripeSceneItem* stripe, PipelineComponent* c, QRectF bbox);
    ~ComponentSceneItem();

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void setIsBeingDragged(bool b);
    void startDrag();
    PipelineComponent* getComponent();

public slots:

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

private:
    QPolygonF myPolygon;  
    ComponentStripeSceneItem* stripe;  
    PipelineComponent* component;  
    QRectF bbox;  
    bool isBeingDragged;  
};

#endif // DESIGNERSCENEITEM_H
