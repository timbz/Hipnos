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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <design/designercontroller.h>
#include <analyse/analysiscontroller.h>
#include <QPropertyAnimation>
#include <QParallelAnimationGroup>

namespace Ui {
    class MainWindow;
}


/**
 * @brief The main application window
 *
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:    
    explicit MainWindow(QWidget *parent = 0);    
    ~MainWindow();

private slots:    
    void on_designButton_clicked();    
    void on_analyseButton_clicked();    
    void on_viewButton_clicked();    
    void on_actionQuit_triggered();    
    void on_actionSave_triggered();    
    void on_actionOpen_triggered();    
    void on_actionAbout_Hipnos_triggered();    
    void on_gaussDataType_clicked();    
    void on_fourierDataType_clicked();

private:
    Ui::MainWindow *ui; 
    DesignerController* designer; 
    AnalysisController* analyser; 

    QParallelAnimationGroup *dataTypeSwitchFrameAnimation; 
    bool validateDesignedPipeline();
    void showDataTypeSwitchFrame();
    void hideDataTypeSwitchFrame();
};

#endif // MAINWINDOW_H
