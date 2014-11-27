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

//#include "inputcontroller.h"

//InputController::InputController(irr::scene::ICameraSceneNode *cam, irr::IrrlichtDevice *device)
//{
//    m_state = S_FREE;
//    m_camera = cam;
//    m_device = device;
//    m_then =  m_device->getTimer()->getTime();
//    m_prevMousePosition = m_device->getCursorControl()->getRelativePosition();
//    m_rotationSpeed = 100.0f;
//    m_goingForward = 0;
//    m_goingBack = 0;
//    m_goingLeft = 0;
//    m_goingRight = 0;
//    m_goingUp = 0;
//    m_goingDown = 0;
//    m_velocity.set(0.0f,0.0f,0.0f);
//}

//bool InputController::OnEvent(const irr::SEvent &event)
//{
//    // Handle mouse events
//    if (event.EventType == irr::EET_MOUSE_INPUT_EVENT)
//    {
//        //if (event.MouseInput.isRightPressed())
//        // enter camera rotation mode
//        if(event.MouseInput.Event == irr::EMIE_RMOUSE_PRESSED_DOWN && m_state != S_CAMERA_ROTATION)
//        {
//            m_state = S_CAMERA_ROTATION;
//            m_device->getCursorControl()->setVisible(false);
//        }
//        // enter free cursor mode
//        else if(event.MouseInput.Event == irr::EMIE_RMOUSE_LEFT_UP && m_state != S_FREE)
//        {
//            // show pointer
//            m_state = S_FREE;
//            m_device->getCursorControl()->setVisible(true);
//        }
//    }
//    // Handle key events
//    else if(event.EventType == irr::EET_KEY_INPUT_EVENT)
//    {
//        if(event.KeyInput.PressedDown)
//        {
//            if(event.KeyInput.Key == irr::KEY_KEY_W || event.KeyInput.Key == irr::KEY_UP)
//            {
//                m_goingForward = true;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_A || event.KeyInput.Key == irr::KEY_LEFT)
//            {
//                m_goingLeft = true;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_S || event.KeyInput.Key == irr::KEY_DOWN)
//            {
//                m_goingBack = true;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_D || event.KeyInput.Key == irr::KEY_RIGHT)
//            {
//                m_goingRight = true;
//            }
//        }
//        else
//        {
//            if(event.KeyInput.Key == irr::KEY_KEY_W || event.KeyInput.Key == irr::KEY_UP)
//            {
//                m_goingForward = false;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_A || event.KeyInput.Key == irr::KEY_LEFT)
//            {
//                m_goingLeft = false;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_S || event.KeyInput.Key == irr::KEY_DOWN)
//            {
//                m_goingBack = false;
//            }
//            if(event.KeyInput.Key == irr::KEY_KEY_D || event.KeyInput.Key == irr::KEY_RIGHT)
//            {
//                m_goingRight = false;
//            }
//        }
//    }

//    // We can update the camera rotation here instead of in the update function
//    irr::core::vector2d<irr::f32> mouseDiff = m_device->getCursorControl()->getRelativePosition() - m_prevMousePosition;
//    if(m_state == S_CAMERA_ROTATION)
//    {
//        //m_device->getCursorControl()->setPosition(m_prevMousePosition);
//        // we get the vector the camera is pointig at
//        irr::core::vector3df direction = (m_camera->getTarget() - m_camera->getAbsolutePosition());
//        // we get the rotation needed to obtain the taget vector from the up vector
//        irr::core::vector3df relativeRotation = direction.getHorizontalAngle();
//        relativeRotation.Y += mouseDiff.X * m_rotationSpeed;
//        relativeRotation.X += mouseDiff.Y * m_rotationSpeed;
//        // set the target to up
//        direction.set(0,0, 1.0f);
//        // transform
//        irr::core::matrix4 mat;
//        mat.setRotationDegrees(irr::core::vector3df(relativeRotation.X, relativeRotation.Y, 0));
//        mat.transformVect(direction);
//        // set
//        m_camera->setTarget(direction + m_camera->getAbsolutePosition());
//    }
//    m_prevMousePosition = m_device->getCursorControl()->getRelativePosition();

//    return false;
//}

//void InputController::update(irr::u32 now)
//{
//    irr::u32 timeDiff = now - m_then;
//    // build acceleration vector
//    irr::core::vector3df accel(0.0f, 0.0f, 0.0f);
//    irr::core::vector3df direction = (m_camera->getTarget() - m_camera->getAbsolutePosition()).normalize();
//    if(m_goingForward) accel += direction * timeDiff * .1f;
//    if(m_goingBack) accel -= direction * timeDiff * .1f;
//    //m_camera->setPosition(m_camera->getPosition() + accel);

//    m_then = now;
//}
