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

#ifndef ANALYSISSLIDERSCENE_H
#define ANALYSISSLIDERSCENE_H

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsSceneMouseEvent>
#include <QWheelEvent>
#include <QGraphicsProxyWidget>

#include "common/pipeline.h"
#include "common/pipelinecomponent.h"
#include "slidersceneitem.h"
#include "chartsceneitem.h"
#include "analysiscomponentsceneitem.h"

/**
 * @brief QGraphicsScene implementationt that draws the optical pipeline and the slider
 *
 */
class AnalysisSliderScene : public QGraphicsScene
{
    Q_OBJECT

public:
    explicit AnalysisSliderScene(QGraphicsView* sv, QObject *parent = 0);

    void drawPipeline(Pipeline* p);
    void clear();
    qreal getSliderValue();
    void zoomOut();
    void zoomIn();
    void updatePipeline(PipelineDataType pipelineComputationDataType);
    void markForUpdate();
    QImage toImage();
    qreal getXScale();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent);
    void wheelEvent(QWheelEvent *event);

signals:
    void sliderMoved(double z);

public slots:
    void setXScale(double s);

private:
    void updateComponentPos();
    void scaleView(qreal scaleFactor);

    SliderSceneItem* sliderItem;
    ChartSceneItem* chartItem;
    QList<AnalysisComponentSceneItem*> componentItems;
    bool dragging;
    QGraphicsView* view;
    QPointF dragginOffset;
    qreal xscale;
    Pipeline* pipeline;
    double scaledPipelineLength;
    qreal componentIconSize;
    QGraphicsProxyWidget *loadingItem;
    bool updateFlag;
};

#endif // ANALYSISSLIDERSCENE_H
