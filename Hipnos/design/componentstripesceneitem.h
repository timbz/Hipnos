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

#ifndef PIPELINESTRIPESCENEITEM_H
#define PIPELINESTRIPESCENEITEM_H

#include <QGraphicsObject>
#include <QPainter>
#include <QGraphicsSceneDragDropEvent>
#include "common/pipelinecomponent.h"
#include "componentsceneitem.h"

/**
 * @brief Implementation of a QGraphicsObject that draws a list of consecutive ComponentListWidgetItem to the DesignerScene
 *
 */
class ComponentStripeSceneItem : public QGraphicsObject
{
    Q_OBJECT

public:
    explicit ComponentStripeSceneItem();
    ~ComponentStripeSceneItem();

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void dropEvent(QGraphicsSceneDragDropEvent * event);
    void dragMoveEvent(QGraphicsSceneDragDropEvent *event);
    bool validateStripe();
    PipelineComponent* buildComponetStripe();
    void autoResizeStripe();
    QVector<ComponentSceneItem*>& getComponetSceneItems();
    int getSlotIdByPosition(QPointF pos);
    void addComponentToSlot(PipelineComponent* c, int slot);
    void addComponentItemToSlot(ComponentSceneItem* item, int slot);
    int getSlotHeight();
    int getSlotDistance();
    void insertSlotAfterSlot(int i);

signals:
    void stripeModified();

public slots:
    void removeItem(ComponentSceneItem* item, bool deleteItem = true);

private:
    QVector<ComponentSceneItem*> componentSlots;  
    int slotWidth;  
    int slotHeight;  
    int padding;  
    QPixmap* background;  
};

#endif // PIPELINESTRIPESCENEITEM_H
