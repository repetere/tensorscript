# TensorScript - Machine Learning and Neural Networks with Tensorflow

[![Coverage Status](https://coveralls.io/repos/github/repetere/tensorscript/badge.svg?branch=master)](https://coveralls.io/github/repetere/tensorscript?branch=master) [![Build Status](https://travis-ci.org/repetere/tensorscript.svg?branch=master)](https://travis-ci.org/repetere/tensorscript)

## Introduction

This library is a compilation of model building modules with a consistent API for quickly implementing Tensorflow at edge(browser) or any JavaScript environment (Node JS / GPU).

### [Read the manual](https://repetere.github.io/tensorscript/manual/overview.html)

## List of Tensorflow models

### Classification

* Deep Learning Classification: [`DeepLearningClassification`](https://repetere.github.io/tensorscript/manual/usage.html#classification)
* Logistic Regression: [`LogisticRegression`](https://repetere.github.io/tensorscript/manual/usage.html#classification)


### Regression

* Deep Learning Regression: [`DeepLearningRegression`](https://repetere.github.io/tensorscript/manual/usage.html#regression)
* Multivariate Linear Regression: [`MultipleLinearRegression`](https://repetere.github.io/tensorscript/manual/usage.html#regression)

### Artificial neural networks (ANN)

* Multi-Layered Perceptrons: [`BaseNeuralNetwork`](https://repetere.github.io/tensorscript/manual/usage.html#neural-networks)

### LSTM Time Series

* Long Short Term Memory Time Series: [`LSTMTimeSeries`](https://repetere.github.io/tensorscript/manual/usage.html#timeseries)
* Long Short Term Memory Multivariate Time Series: [`LSTMMultivariateTimeSeries`](https://repetere.github.io/tensorscript/manual/usage.html#timeseries)

## Basic Usage

TensorScript is and ECMA Script module designed to be used in an `ES2015+` environment, if you need compiled modules for older versions of node use the compiled modules in the bundle folder.

Please read more on tensorflow configuration options, specifying epochs, and using custom layers in [configuration](https://repetere.github.io/tensorscript/manual/overview.html#configuration).

### Regression Examples

```javascript
import { MultipleLinearRegression, DeepLearningRegression, } from 'tensorscript';
import ms from 'modelscript';

async function main(){
  const independentVariables = [ 'sqft', 'bedrooms',];
  const dependentVariables = [ 'price', ];
  const housingdataCSV = await ms.csv.loadCSV('./test/mock/data/portland_housing_data.csv');
  const DataSet = new ms.DataSet(housingdataCSV);
  const x_matrix = DataSet.columnMatrix(independentVariables);
  const y_matrix = DataSet.columnMatrix(dependentVariables);
  const MLR = new MultipleLinearRegression();
  await MLR.train(x_matrix, y_matrix);
  const DLR = new DeepLearningRegression();
  await DLR.train(x_matrix, y_matrix);
  //1600 sqft, 3 bedrooms
  await MLR.predict([1650,3]); //=>[293081.46]
  await DLR.predict([1650,3]); //=>[293081.46]
}
main();
```

### Classification Examples

```javascript
import { DeepLearningClassification, } from 'tensorscript';
import ms from 'modelscript';

async function main(){
  const independentVariables = [
    'sepal_length_cm',
    'sepal_width_cm',
    'petal_length_cm',
    'petal_width_cm',
  ];
  const dependentVariables = [
    'plant_Iris-setosa',
    'plant_Iris-versicolor',
    'plant_Iris-virginica',
  ];
  const housingdataCSV = await ms.csv.loadCSV('./test/mock/data/iris_data.csv');
  const DataSet = new ms.DataSet(housingdataCSV).fitColumns({ columns: {plant:'onehot'}, });
  const x_matrix = DataSet.columnMatrix(independentVariables);
  const y_matrix = DataSet.columnMatrix(dependentVariables);
  const nnClassification = new DeepLearningClassification();
  await nnClassification.train(x_matrix, y_matrix);
  const input_x = [
    [5.1, 3.5, 1.4, 0.2, ],
    [6.3, 3.3, 6.0, 2.5, ],
    [5.6, 3.0, 4.5, 1.5, ],
    [5.0, 3.2, 1.2, 0.2, ],
    [4.5, 2.3, 1.3, 0.3, ],
  ];
  const predictions = await nnClassification.predict(input_x); 
  const answers = await nnClassification.predict(input_x, { probability:false, });
  /*
    predictions = [
      [ 0.989512026309967, 0.010471616871654987, 0.00001649192017794121, ],
      [ 0.0000016141033256644732, 0.054614484310150146, 0.9453839063644409, ],
      [ 0.001930746017023921, 0.6456733345985413, 0.3523959517478943, ],
      [ 0.9875779747962952, 0.01239941269159317, 0.00002274810685776174, ],
      [ 0.9545140862464905, 0.04520365223288536, 0.0002823179238475859, ],
    ];
    answers = [
      [ 1, 0, 0, ], //setosa
      [ 0, 0, 1, ], //virginica
      [ 0, 1, 0, ], //versicolor
      [ 1, 0, 0, ], //setosa
      [ 1, 0, 0, ], //setosa
    ];
   */
}
main();
```

```javascript
import { LogisticRegression, } from 'tensorscript';
import ms from 'modelscript';

async function main(){
  const independentVariables = [
    'Age',
    'EstimatedSalary',
  ];
  const dependentVariables = [
    'Purchased',
  ];
  const housingdataCSV = await ms.csv.loadCSV('./test/mock/data/social_network_ads.csv');
  const DataSet = new ms.DataSet(housingdataCSV).fitColumns({ columns: {Age:['scale','standard'],
  EstimatedSalary:['scale','standard'],}, });
  const x_matrix = DataSet.columnMatrix(independentVariables);
  const y_matrix = DataSet.columnMatrix(dependentVariables);
  const LR = new LogisticRegression();
  await LR.train(x_matrix, y_matrix);
  const input_x = [
    [-0.062482849427819266, 0.30083326827486173,], //0
    [0.7960601198093905, -1.1069168538010206,], //1
    [0.7960601198093905, 0.12486450301537644,], //0
    [0.4144854668150751, -0.49102617539282206,], //0
    [0.3190918035664962, 0.5061301610775946,], //1
  ];
  const predictions = await LR.predict(input_x); // => [ [ 0 ], [ 0 ], [ 1 ], [ 0 ], [ 1 ] ];
}
main();
```

### Time Series Example

```javascript
import { LSTMTimeSeries, } from 'tensorscript';
import ms from 'modelscript';

async function main(){
  const dependentVariables = [
    'Passengers',
  ];
  const airlineCSV = await ms.csv.loadCSV('./test/mock/data/airline-sales.csv');
  const DataSet = new ms.DataSet(airlineCSV);
  const x_matrix = DataSet.columnMatrix(independentVariables);
  const TS = new LSTMTimeSeries();
  await TS.train(x_matrix);
  const forecastData = TS.getTimeseriesDataSet([ [100 ], [200], [300], ])
  await TS.predict(forecastData.x_matrix); //=>[200,300,400]
}
main();
```

### Testing

```sh
$ npm i
$ npm test
```

### Contributing

Fork, write tests and create a pull request!

### Misc

As of Node 8, ES modules are still used behind a flag, when running natively as an ES module

```sh
$ node --experimental-modules manual/examples/ex_regression-boston.mjs
# Also there are native bindings that require Python 2.x, make sure if you're using Anaconda, you build with your Python 2.x bin
$ npm i --python=/usr/bin/python
 ```

 ### Special Thanks
 - [Machine Learning Mastery](https://machinelearningmastery.com/)
 - [Super Data Science](https://www.superdatascience.com/)
 - [Python Programming](https://pythonprogramming.net/)
 - [Towards Data Science](https://towardsdatascience.com/)
 - [ml.js](https://github.com/mljs/ml)

License
----

MIT