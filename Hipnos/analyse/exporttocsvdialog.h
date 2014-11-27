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

#ifndef EXPORTTOCSVDIALOG_H
#define EXPORTTOCSVDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QRadioButton>
#include <QSpinBox>
#include "common/pipeline.h"

/**
 * @brief QDialog implementation that gives the user the posibility to export a given simulation result to a CSV file
 *
 */
class ExportToCsvDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ExportToCsvDialog(Pipeline* p, PipelineDataType dt, double z, QWidget *parent = 0);
    
signals:
    
public slots:
    void onChooseFile();
    void exportToCsv();

private:
    void showErrorAndAbort(QString error);

    Pipeline* pipeline;  
    PipelineDataType dataType;  
    double zValue;  

    QLineEdit* fileNameEdit;  

    QRadioButton* realButton;  
    QRadioButton* imagButton;  
    QRadioButton* absButton;  
    QRadioButton* argButton;  

    QRadioButton* normalButton;  
    QRadioButton* scientificButton;  

    QLineEdit* delimiterEdit;  

    QSpinBox* dataPrecisionSpinBox;  
};

#endif // EXPORTTOCSVDIALOG_H
