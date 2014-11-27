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

#ifndef SPINBOX_H
#define SPINBOX_H

#include <QSpinBox>

/**
 * @brief QValidator used to validate even numbers
 *
 */
class EvenIntValidator : public QValidator
{
    State validate(QString &input, int &pos ) const
    {
        bool ok;
        int num = input.toInt(&ok);
        if(!ok)
            return Invalid;
        if(num % 2 == 0)
            return Acceptable;
        else
            return Invalid;
    }
};


/**
 * @brief Extends the functionality of QSpinBox by adding a QValidator
 *
 */
class SpinBox : public QSpinBox
{

public:
    SpinBox();
    void setValidator(QValidator* v);

protected:
    QValidator::State validate(QString &text, int &pos) const;

private:
    QValidator* validator;
};

#endif // SPINBOX_H
