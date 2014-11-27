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

#ifndef OPENCLPLATFORMANDDEVICEDIALOG_H
#define OPENCLPLATFORMANDDEVICEDIALOG_H

#include <QDialog>
#include <QRadioButton>
#include <QPushButton>
#include <QHash>

#include "opencldevice.h"

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class OpenCLPlatformAndDeviceDialog : public QDialog
{
    Q_OBJECT
public:
    /**
     * @brief
     *
     * @param devices
     */
    /**
     * @brief
     *
     * @param devices
     */
    explicit OpenCLPlatformAndDeviceDialog(QList<OpenCLDevice> devices);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    OpenCLDevice getSelectedDevice();
    /**
     * @brief
     *
     * @param devices
     */
    /**
     * @brief
     *
     * @param devices
     */
    static OpenCLDevice selectDevice(QList<OpenCLDevice> devices);
signals:

public slots:

private:
    QPushButton* okButton;  
    QHash<QRadioButton*, OpenCLDevice> radioButtons;  
};

#endif // OPENCLPLATFORMANDDEVICEDIALOG_H
