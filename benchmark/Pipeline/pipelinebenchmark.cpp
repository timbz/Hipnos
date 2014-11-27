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

#include "pipelinebenchmark.h"
#include "common/components/gaussianbeamsourcecomponent.h"
#include "common/components/aperturecomponent.h"
#include "common/components/propagationcomponent.h"
#include "common/components/thinlenscomponent.h"
#include "common/hipnossettings.h"
#include "common/pipeline.h"

int isprim(unsigned int z)
{
    unsigned int i;
    for(i = 2; i <= z/2; i++)
    {
        if(!(z % i))
        {
            return 0;
        }
    }
    return 1;
}

QSet<int> primfac(unsigned int z)
{
    QSet<int> r;
    unsigned int i;
    while(z > 1)
    {
        i = 2;
        while(1)
        {
            if(!(z % i) && isprim(i))
            {
                r << i;
                z /= i;
                break;
            }
            i++;
        }
    }
    return r;
}

PipelineBenchmark::PipelineBenchmark()
{
    HipnosSettings::getInstance().setUsePipelineCash(true);
    Math::init();
    plugins = Math::getAvailablePlugins();
}

PipelineBenchmark::~PipelineBenchmark()
{
    Math::destroy();
}

void PipelineBenchmark::testFourierPropagation_data()
{
    QTest::addColumn<int>("pluginIndex");
    QTest::addColumn<int>("matrixSize");    
    for(int i = 0; i < plugins.size(); i ++)
    {
        MathPlugin* p = plugins[i];
        if(p->getName().contains("APPML"))
        {
            for(int matrixSize = 2; matrixSize <= 502; matrixSize += 2)
            {
                QSet<int> fac = primfac(matrixSize);
                fac.remove(2);
                fac.remove(3);
                fac.remove(5);
                if(fac.isEmpty())
                {
                    QString name = "plugin=[" + p->getName() + "] matrixSize=[" + QString::number(matrixSize) + "]";
                    QTest::newRow(name.toStdString().c_str()) << i << matrixSize;
                }
            }
        }
    }
    for(int matrixSize = 2; matrixSize <= 502; matrixSize += 10)
    {
        for(int i = 0; i < plugins.size(); i++)
        {
            MathPlugin* p = plugins[i];
            if(p->getName().contains("APPML"))
                continue;
            QString name = "plugin=[" + p->getName() + "] matrixSize=[" + QString::number(matrixSize) + "]";
            QTest::newRow(name.toStdString().c_str()) << i << matrixSize;
        }
    }
}

void PipelineBenchmark::testFourierPropagation()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);

    Math::setActivePlugin(plugins[pluginIndex]);

    double lambda = 1030 * 0.000000001; //m
    double W0 = 0.002; //m
    double E0 = 1;
    int phaseradius = 10;

    GaussianBeamSourceComponent source(Spectrum(lambda), W0, phaseradius, E0, matrixSize);
    PropagationComponent drift(100, PropagationComponent::PM_NEAR_FIELD);

    PipelineConnection con(&source, &drift);
    source.setOutputConnection(&con);
    drift.setInputConnection(&con);
    PipelineSink sink(&drift);
    drift.setOutputConnection(&sink);

    // cache
    PipelineSpectrumData* spectrum = source.getOutputConnection()->getData(DT_FOURIER);
    delete spectrum;
    QBENCHMARK {
        spectrum = sink.getData(DT_FOURIER);
        delete spectrum;
    }
}
QTEST_APPLESS_MAIN(PipelineBenchmark)
