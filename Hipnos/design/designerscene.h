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

#ifndef DESIGNERSCENE_H
#define DESIGNERSCENE_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneDragDropEvent>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include "componentsceneitem.h"
#include "common/pipeline.h"
#include "componentstripesceneitem.h"
#include "common/components/groupcomponent.h"

class DesignerController;

/**
 * @brief Implementation of a QGraphicsScene used to display the current Pipeline in the designer view
 *
 */
class DesignerScene : public QGraphicsScene
{
public:
    explicit DesignerScene(DesignerController* c);

    ComponentSceneItem* getDraggedItem();
    void setDraggedItem(ComponentSceneItem* item);
    Pipeline* getPipeline();
    void deleteSelectedItems();
    GroupComponent* groupSelectedItems();
    void ungroupSelectedItems();
    ComponentStripeSceneItem* getStripe();
    void reset();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);

private:
    QGraphicsView* graphicView;  
    ComponentSceneItem* draggedItem;  
    ComponentStripeSceneItem* stripe ;  
    bool isDragging;  
};

#endif // DESIGNERSCENE_H
