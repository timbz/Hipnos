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

#ifndef THREADSAFEQUEUE_H
#define THREADSAFEQUEUE_H

#include <QMutex>
#include <QMutexLocker>
#include <QQueue>

/**
 * @brief QMutex locked QQueue
 *
 */
template <class T> class ThreadSafeQueue
{

private:
    QQueue<T> queue;  
    QMutex lock;  

public:
    /**
     * @brief Removes the head item in the queue and returns it. This function assumes that the queue isn't empty.
     *
     * @return T
     */
    T dequeue()
    {
        QMutexLocker locker(&lock);
        return queue.dequeue();
    }

    /**
     * @brief Adds value t to the tail of the queue.
     *
     * @param t
     */
    void enqueue(const T& t)
    {
        QMutexLocker locker(&lock);
        queue.enqueue(t);
    }

    /**
     * @brief Returns true if the list contains no items; otherwise returns false.
     *
     * @return bool
     */
    bool isEmpty()
    {
        QMutexLocker locker(&lock);
        return queue.isEmpty();
    }
};

#endif // THREADSAFEQUEUE_H
