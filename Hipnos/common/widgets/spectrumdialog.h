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

#ifndef SPECTRUMDIALOG_H
#define SPECTRUMDIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QDoubleValidator>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>

#include "qcustomplot.h"
#include "common/spectrum.h"


/**
 * @brief QDialog used in a SpectrumDialog to generate a default Spectrum
 *
 */
class LoadPresetDialog : public QDialog
{
    Q_OBJECT

public:
    LoadPresetDialog(Spectrum *s, QWidget* p = 0);

public slots:
    void onOkClicked();

private:
    Spectrum* spectrum;
    QSpinBox* resolution;
    QDoubleSpinBox* center;
    QDoubleSpinBox* width;
    QComboBox* funtion;
};

/**
 * @brief QDialog used to edit a Spectrum
 *
 */
class SpectrumDialog : public QDialog
{
    Q_OBJECT

public:
    SpectrumDialog(Spectrum s);

    Spectrum getSpectrum();

public slots:
    void onTableCellChanged(int i, int j);
    void onTableCellClicked(int i, int j);
    void onLoadPresetClicked();
    void onLoadCSVClicked();

private:
    void updateTable();
    void updatePlot();

    QDoubleValidator* validator;
    Spectrum spectrum;
    QTableWidget* table;
    QCustomPlot* plot;
};

#endif // SPECTRUMDIALOG_H
