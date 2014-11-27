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

#ifndef ANALYSISCONTROLLER_H
#define ANALYSISCONTROLLER_H

#include <QObject>
#include <QThread>
#include <QVTKWidget.h>
#include <QGraphicsView>
#include <QVBoxLayout>
#include <QLabel>
#include <QMutex>

#include "common/pipelinedata.h"
#include "common/pipeline.h"
#include "analysissliderscene.h"
#include "componentpropertywidget.h"
#include "vtkviewpropertywidget.h"
#include "vtkpipeline.h"
#include "ui_mainwindow.h"

/**
 * @brief Central class of the analysis view. It handels events between widgets, starts the simulation pipeline and talks to VTK
 *
 */
class AnalysisController : public QObject
{
    Q_OBJECT

public:

    /**
     * @brief QThread implementation that runs the simulation pipeline
     *
     */
    class PropagationThread : public QThread
    {
    public:
        PropagationThread(AnalysisController* c, double z);
        void run();

    private:
        double zValue;
        AnalysisController* controller;
    };

    AnalysisController(Ui::MainWindow* ui);
    ~AnalysisController();

    void setPipeline(Pipeline* p);
    void computePropagationAt(double z);
    void createGausPropagationThread(double z);

signals:

public slots:
    void onSliderChanged(double z);
    void onPropertyChanged(PipelineComponent::Property p);
    void onFrequencyPercentageRangeChanged(int min, int max);
    void onThreadFinished();
    void onSliderSceneSelectionChanged();
    void exportVtkAsJpeg();
    void exportSliderSceneAsJpeg();
    void exportToCsv();
    void sliderSceneZoomIn();
    void sliderSceneZoomOut();
    void setPipelineComputationDataType(PipelineDataType dataType);

private:
    QVTKWidget* vtkWidget;
    QLabel* sliderPosLabel;
    Pipeline* pipeline;
    AnalysisSliderScene* sliderScene;
    VtkViewPropertyWidget* vtkViewPropertyWidget;
    ComponentPropertyWidget* componentPropertyWidget;
    VtkPipeline* vtkPipeline;
    QLabel* loadingAnimation;

    QThread* currentThread;
    QThread* nextThread;
    PipelineDataType pipelineComputationDataType;
};

#endif // ANALYSISCONTROLLER_H
