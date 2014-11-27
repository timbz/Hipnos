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

#ifndef AXISGRAPHICSITEM_H
#define AXISGRAPHICSITEM_H

#include <QGraphicsItem>
#include <QPainter>
#include <QThread>
#include <QGraphicsProxyWidget>
#include "common/pipeline.h"

/**
 * @brief This QGraphicsObject draws the beam width to the AnalysisSliderScene when the pipeline uses Gaussian simulation
 *
 */
class ChartSceneItem : public QGraphicsObject
{
    Q_OBJECT

public:
    explicit ChartSceneItem(QGraphicsProxyWidget* la);
    ~ChartSceneItem();

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    void setPipeline(Pipeline* p);
    void updateChart(PipelineDataType dataType);
    double getXScale();
    double getStepSize();
    double getYAxisSize();

public slots:
    void setXScale(double s);
    void setStepSize(double s);
    void setYAxisSize(double s);

private:
    Pipeline* pipeline;
    double pipelineLength;

    double stepSize;
    double xscale;
    double yAxisSize;

    QSizeF arrowSize;
    qreal axisPadding;

    double wYscale;
    QPolygonF wPolygon;
    QList<double> wData;
    double wMax;

    PipelineDataType pipelineComputationDataType;
    QGraphicsProxyWidget* loadingAnim;
};

#endif // AXISGRAPHICSITEM_H
