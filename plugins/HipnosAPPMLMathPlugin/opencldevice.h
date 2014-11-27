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

#ifndef OPENCLDEVICE_H
#define OPENCLDEVICE_H

#include <clAmdBlas.h>
#include <QString>

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
enum OCLDeviceType
{
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_GPU,
    DEVICE_TYPE_ACCELERATOR,
    DEVICE_TYPE_DEFAULT
};

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
struct OpenCLDevice
{
    cl_platform_id PlatformId;  
    QString PlatformName;  
    cl_device_id DeviceId;  
    QString DeviceName;  
    QString DeviceVersion;  
    QString DeviceExtensions;  
    OCLDeviceType DeviceType;  

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
    OpenCLDevice()
    {
    }

/**
 * @brief
 *
 * @param platformId
 * @param platformName
 * @param deviceId
 * @param deviceName
 * @param deviceType
 * @param deviceVersion
 * @param deviceExtensions
 */
/**
 * @brief
 *
 * @param platformId
 * @param platformName
 * @param deviceId
 * @param deviceName
 * @param deviceType
 * @param deviceVersion
 * @param deviceExtensions
 */
    OpenCLDevice(cl_platform_id platformId,
                 QString platformName,
                 cl_device_id deviceId,
                 QString deviceName,
                 OCLDeviceType deviceType,
                 QString deviceVersion,
                 QString deviceExtensions)
    {
        PlatformId = platformId;
        PlatformName = platformName;
        DeviceId = deviceId;
        DeviceName = deviceName;
        DeviceType = deviceType;
        DeviceVersion = deviceVersion;
        DeviceExtensions = deviceExtensions;
    }

    /**
     * @brief
     *
     * @return bool
     */
    /**
     * @brief
     *
     * @return bool
     */
    bool supportsDoublePrecision()
    {
        return DeviceExtensions.contains("cl_khr_fp64");
    }
};

#endif // OPENCLDEVICE_H
