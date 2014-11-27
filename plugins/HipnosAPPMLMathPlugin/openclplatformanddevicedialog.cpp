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

#include "openclplatformanddevicedialog.h"
#include "common/hipnossettings.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QApplication>
#include <QSettings>
#include <QDebug>

#include <iostream>

OpenCLPlatformAndDeviceDialog::OpenCLPlatformAndDeviceDialog(QList<OpenCLDevice> devices) :
    QDialog(0)
{
    setWindowIcon(QIcon(":/icons/app-icon.png"));

    QVBoxLayout *layout = new QVBoxLayout;

    QLabel* title = new QLabel("<b>Select a device for the APPML Math Plugin:</b>");
    layout->addWidget(title);

    layout->addSpacing(20);

    // we preselect the GPU with the longest extension string (should be the better one)
    int maxExtensionLenght = -1;
    QString currentPlatform("");
    foreach (OpenCLDevice device, devices)
    {
        if(currentPlatform != device.PlatformName)
        {
            currentPlatform = device.PlatformName;
            layout->addWidget(new QLabel(currentPlatform));
        }
        QRadioButton* r = new QRadioButton(device.DeviceName + "(" + device.DeviceVersion + ")");
        layout->addWidget(r);
        radioButtons[r] = device;

        // default selection
        if(device.DeviceType == DEVICE_TYPE_GPU &&
                device.DeviceExtensions.length() > maxExtensionLenght)
        {
            maxExtensionLenght = device.DeviceExtensions.length();
            r->setChecked(true);
        }
    }

    // if no GPU select first device (probably CPU)
    if(maxExtensionLenght < 0)
    {
        radioButtons.keys().first()->setChecked(true);
    }

    layout->addSpacing(20);

    okButton = new QPushButton("OK");
    connect(okButton,SIGNAL(clicked()),this,SLOT(accept()));
    layout->addWidget(okButton);

    setLayout(layout);
}

OpenCLDevice OpenCLPlatformAndDeviceDialog::getSelectedDevice()
{
    foreach(QRadioButton* r, radioButtons.keys())
    {
        if(r->isChecked())
        {
            return radioButtons[r];
        }
    }
    return radioButtons[0];
}

OpenCLDevice OpenCLPlatformAndDeviceDialog::selectDevice(QList<OpenCLDevice> devices)
{
    // if we only have one device, we return it
    if(devices.size() == 1)
    {
        return devices.first();
    }
    // check in QStettings
    QString platformName = HipnosSettings::getInstance().value("Plugins/AppmlMathPlugin/platform").toString();
    QString deviceName = HipnosSettings::getInstance().value("Plugins/AppmlMathPlugin/device").toString();
    foreach (OpenCLDevice device, devices)
    {
        if(platformName == device.PlatformName &&
                deviceName == device.DeviceName)
        {
            return device;
        }
    }

    // ask the user to select a device
    OpenCLDevice result;
    if(qApp) // test we have are in a gui env
    {
        OpenCLPlatformAndDeviceDialog dia(devices);
        dia.exec();
        result = dia.getSelectedDevice();
    }
    else
    {
        std::cout << "Listing devices:" << std::endl;
        QHash<int, OpenCLDevice> selectionsIds;
        int deviceCount = 1;
        QString currentPlatform("");
        foreach (OpenCLDevice device, devices)
        {
            if(currentPlatform != device.PlatformName)
            {
                currentPlatform = device.PlatformName;
                std::cout << std::endl << device.PlatformName.toStdString() << std::endl;
            }
            std::cout << "\t[" << deviceCount << "] " << (device.DeviceName + "(" + device.DeviceVersion + ")").toStdString() << std::endl;
            selectionsIds[deviceCount] = device;
            deviceCount++;
        }
        int s = 0;
        while(!selectionsIds.contains(s))
        {
            std::cout << std::endl << "Select a device " << "[1-" << deviceCount-1 << "]:";
            std::cin >> s;
        }

        result = selectionsIds[s];
    }
    // store selection
    HipnosSettings::getInstance().setValue("Plugins/AppmlMathPlugin/platform", result.PlatformName);
    HipnosSettings::getInstance().setValue("Plugins/AppmlMathPlugin/device", result.DeviceName);
    return result;
}
