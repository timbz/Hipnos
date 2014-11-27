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

#ifndef STATUSMESSAGE_H
#define STATUSMESSAGE_H

#include <QStatusBar>

/**
 * @brief A singelton class used to show some status messages to the user
 *
 */
class StatusMessage
{

public:
    /**
     * @brief Returns a reference to the instance
     *
     * @return StatusMessage &
     */
    static StatusMessage& getInstance()
    {
        static StatusMessage instance;
        return instance;
    }

    /**
     * @brief Shows a message to the user
     *
     * @param text
     * @param timeout
     */
    static void show(const QString& text, int timeout = 0)
    {
        getInstance().showMessage(text, timeout);
    }

    /**
     * @brief sets the QStatusBar used to display messages
     *
     * @param b
     */
    void setStatusBar(QStatusBar* b)
    {
        statusBar = b;
    }

private:

    QStatusBar* statusBar;  

    /**
     * @brief Private constructor
     *
     */
    StatusMessage()
    {
        statusBar = 0;
    }

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    StatusMessage(StatusMessage const&);

    /**
     * @brief Not implemented (it' s a singelton)
     *
     * @param
     */
    void operator=(StatusMessage const&);

    /**
     * @brief Use the public static show() method to display messages
     *
     * @param text
     * @param timeout
     */
    void showMessage(const QString& text, int timeout)
    {
        if(statusBar)
            statusBar->showMessage(text, timeout);
    }

};

#endif // STATUSMESSAGE_H
