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

#ifndef DESIGNERCONTROLLER_H
#define DESIGNERCONTROLLER_H

#include <QObject>
#include <QGraphicsView>
#include <QListWidget>
#include "designerscene.h"
#include "componentsceneitem.h"
#include "common/pipeline.h"
#include "ui_mainwindow.h"

/**
 * @brief Central class of the design view. It handels events between widgets, and builds the Pipeline
 *
 */
class DesignerController : public QObject
{
    Q_OBJECT

public:
    DesignerController(Ui::MainWindow* ui);
    ~DesignerController();

    Pipeline* getPipeline();
    QGraphicsView* getGraphicView();
    void save();
    void load();
    bool getPipelineModified();
signals:

public slots:
    void setPipelineModified();
    void zoomOut();
    void zoomIn();
    void deleteSelectedItems();
    void groupSelectedItems();
    void ungroupSelectedItems();
    void deleteCustomComponent();

private:
    QGraphicsView* view;  
    DesignerScene* scene;  
    QListWidget* baseComponentList;  
    QListWidget* customComponentList;  
    bool pipelineModified;  
    Pipeline* currentPipeline;  
    qreal zoomFactor;  
};

#endif // DESIGNERCONTROLLER_H
