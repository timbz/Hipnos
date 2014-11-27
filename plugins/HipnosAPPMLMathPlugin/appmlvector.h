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

#ifndef APPMLVECTOR_H
#define APPMLVECTOR_H

#include <QString>
#include <clAmdBlas.h>

#include "common/math/vector.h"

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class AppmlVector : public Vector
{
public:
/**
 * @brief
 *
 * @param c
 * @param q
 * @param s
 */
/**
 * @brief
 *
 * @param c
 * @param q
 * @param s
 */
    AppmlVector(cl_context c, cl_command_queue q, int s);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~AppmlVector();

    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void* data();
    /**
     * @brief
     *
     * @param i
     */
    /**
     * @brief
     *
     * @param i
     */
    std::complex<double> get(int i);
    /**
     * @brief
     *
     * @param i
     * @param s
     */
    /**
     * @brief
     *
     * @param i
     * @param s
     */
    void set(int i, std::complex<double> s);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    Vector* clone();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    int getSize();

    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void setDeviceDataChanged();

private:
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void updateHostData();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void updateDeviceData();
    /**
     * @brief
     *
     * @param functionName
     */
    /**
     * @brief
     *
     * @param functionName
     */
    void checkForError(QString functionName);

    std::complex<double>* hostRaw;  
    bool hostDataChanged;  
    cl_mem deviceRaw;  
    bool deviceDataChanged;  
    int size;  
    int err;  
    cl_context ctx;  
    cl_command_queue queue;  
    cl_event event;  
};

#endif // APPMLVECTOR_H
