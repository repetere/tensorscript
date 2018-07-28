# @tensorscript/ts-deeplearning

[![Coverage Status](https://coveralls.io/repos/github/repetere/ts-deeplearning/badge.svg?branch=master)](https://coveralls.io/github/repetere/ts-deeplearning?branch=master) [![Build Status](https://travis-ci.org/repetere/ts-deeplearning.svg?branch=master)](https://travis-ci.org/repetere/ts-deeplearning)

Deep Learning Classification, Clustering and Regression with Tensorflow
### [Full Documentation](<https://github.com/repetere/ts-deeplearning/blob/master/docs/API.md>)

### Installation

```sh
$ npm i @tensorscript/ts-deeplearning
```

### Usage

Test against the [Portland housing price dataset](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html)

```javascript
import { MultipleLinearRegression, } from '@tensorscript/ts-deeplearning';
import ms from 'modelscript';

function scaleColumnMap(columnName) {
  return {
    name: columnName,
    options: {
      strategy: 'scale',
      scaleOptions: {
        strategy:'standard'
      }
    }
  }
}

async function main(){
  const housingdataCSV = await ms.csv.loadCSV('./test/mock/data/portland_housing_data.csv', {
    colParser: {
      sqft: 'number',
      bedrooms: 'number',
      price: 'number',
    }
  });
  /*
  housingdataCSV = [ 
    { sqft: 2104, bedrooms: 3, price: 399900 },
    { sqft: 1600, bedrooms: 3, price: 329900 },
    ...
    { sqft: 1203, bedrooms: 3, price: 239500 } 
  ] 
  */
  const DataSet = new ms.DataSet(housingdataCSV);
  DataSet.fitColumns({
    columns: [
      'sqft',
      'bedrooms',
      'price',
    ].map(scaleColumnMap),
    returnData:true,
  });
  const independentVariables = [ 'sqft', 'bedrooms',];
  const dependentVariables = [ 'price', ];
  const x_matrix = DataSet.columnMatrix(independentVariables); 
  const y_matrix = DataSet.columnMatrix(dependentVariables);
  /* x_matrix = [
      [2014, 3],
      [1600, 3],
    ];
    y_matrix = [
      [399900],
      [329900],
    ];
    const y_vector = ms.util.pivotVector(y_matrix)[ 0 ];// not used but just illustrative
    // y_vector = [ 399900, 329900]
   */
  const testSqft = DataSet.scalers.get('sqft').scale(1650);
  const testBedrooms = DataSet.scalers.get('bedrooms').scale(3);
  const input_x = [
    testSqft,
    testBedrooms,
  ]; // input_x: [ -0.4412732005944351, -0.2236751871685913 ]
  const tfMLR = new MultipleLinearRegression();
  const model = await tfMLR.train(x_matrix, y_matrix);
  const scaledPrediction = await tfMLR.predict(input_x); // [ -0.3785287367962629 ]
  const prediction = DataSet.scalers.get('price').descale(scaledPrediction); // prediction: 293081.4643348962
}

main();

```

This MLR module give you a similar ml.js interface for machine learning

```javascript
// Similarly with ml.js
import ms from 'modelscript';
const MLR = ms.ml.Regression.MultivariateLinearRegression;
const regression = new MLR(x_matrix, y_matrix);
const MLJSscaledPrediction = regression.predict(input_x); //[ -0.3785287367962629 ],
const MLJSprediction = DataSet.scalers.get('price').descale(MLJSscaledPrediction); // prediction: 293081.4643348962
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
$ node --experimental-modules my-machine-learning-script.mjs
# Also there are native bindings that require Python 2.x, make sure if you're using Andaconda, you build with your Python 2.x bin
$ npm i --python=/usr/bin/python
 ```

License
----

MIT