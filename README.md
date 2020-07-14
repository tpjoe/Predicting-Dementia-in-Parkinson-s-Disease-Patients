# Predicting-Dementia-in-Parkinson-s-Disease-Patients
This is a source code for producing results in the "Predicting Dementia in Parkinson's Disease Patients" accepted to npj Parkinson's Disease

**Note that to run these scripts you would need the data, which can be obtained upon contacting us (bchol@stanford.edu) due to privacy issues**

Once obtain the data, the following scripts would reproduce results as follow:
**1. data_preprocessing.ipynb** - This notebook will have to be run first to generate processed data for other downstream machine learning models and analyses. This module is in Python and the data is generated for any application in R and in MATLAB (multitask learning). \
**2. multilevel_models.ipynb** - This notebook operates in R and contains all mix-effect models prediction and analyses found in the manuscript. \
**3. multitask.ipynb** - This module analyzes and plots results from multitask learning model. The multitask model results can be obtained by running the script \ **scripts/TemporalLASSO.py**, which would require the user to setup MongoDB, pyMongo, and MATLAB path in their machine. \
**4. survival.ipynb** - This notebook contain the source code for plotting interested effects on survival, written in R language. \
**5. Sankey.ipynb** - This notebook plots the Sankey diagram of the cohort, it requires a plotly API. \



The MIT License (MIT) \
Copyright (c) 2020 \

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: \

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. \

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
