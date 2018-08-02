/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_nn-portland.mjs
 */
import * as ms from 'modelscript'; // used for scaling, data manipulation
import { BaseNeuralNetwork, } from '../../index.mjs';
//if running on node
// import tf from '@tensorflow/tfjs-node';

const independentVariables = [
  'CRIM',
  'ZN',
  'INDUS',
  'CHAS',
  'NOX',
  'RM',
  'AGE',
  'DIS',
  'RAD',
  'TAX',
  'PTRATIO',
  'B',
  'LSTAT',
];
const dependentVariables = [
  'MEDV',
];
const columns = independentVariables.concat(dependentVariables);

function scaleColumnMap(columnName) {
  return {
    name: columnName,
    options: {
      strategy: 'scale',
      scaleOptions: {
        strategy:'standard',
      },
    },
  };
}

async function main() {
  /*
  CSVData = [
    { CRIM: 0.00632, ZN: 18, INDUS: 2.31, CHAS: 0, NOX: 0.538, RM: 6.575, AGE: 65.2, DIS: 4.09, RAD: 1, TAX: 296, PTRATIO: 15.3, B: 396.9, LSTAT: 4.98, MEDV: 24 },
    { CRIM: 0.02731, ZN: 0, INDUS: 7.07, CHAS: 0, NOX: 0.469, RM: 6.421, AGE: 78.9, DIS: 4.9671, RAD: 2, TAX: 242, PTRATIO: 17.8, B: 396.9, LSTAT: 9.14, MEDV: 21.6 },
    ...
  ]
  */
  const CSVData = await ms.csv.loadCSV('./test/mock/data/boston_housing_data.csv');
  const CSVDataSet = new ms.DataSet(CSVData);
  CSVDataSet.fitColumns({
    columns: columns.map(scaleColumnMap),
  });
  // console.log(CSVDataSet.scalers)
  const testTrainSplit = ms.cross_validation.train_test_split(CSVDataSet.data, { train_size: 0.7, });
  const { train, test, } = testTrainSplit;
  const testDataSet = new ms.DataSet(test);
  const trainDataSet = new ms.DataSet(train);
  const x_matrix_train = trainDataSet.columnMatrix(independentVariables);
  const y_matrix_train = trainDataSet.columnMatrix(dependentVariables);
  const x_matrix_test = testDataSet.columnMatrix(independentVariables); 
  const y_matrix_test = testDataSet.columnMatrix(dependentVariables);
  /* 
  x_train = [
    [ -0.41936692921321594, 0.2845482693404666, -1.2866362317172035, -0.272329067679207, -0.1440748547324509, 0.4132629204530747, -0.119894767215809, 0.1400749839795629, -0.981871187861867, -0.6659491794887338, -1.457557967289609, 0.4406158949991029, -1.074498970343932 ],
    [ -0.41692666996409716, -0.4872401872268264, -0.5927943782429392, -0.272329067679207, -0.7395303607434242, 0.1940823874370036, 0.3668034264326209, 0.5566090495704026, -0.8670244885881488, -0.9863533804386945, -0.3027944997494681, 0.4406158949991029, -0.49195252491856634 ]
    ...
  ]
  y_matrix_test = [ 
    [ 0.15952778852449556 ],
    [ -0.1014239172731213 ],
    ...
  ]
  */
  const fit = {
    epochs: 100,
    batchSize:10,
  };
  const compile = {
    loss: 'meanSquaredError',
    optimizer: 'adam',
  };
  const nnLR = new BaseNeuralNetwork({
    layerPreference: 'deep',
    compile,
    fit,
  }, {
    // tf - can switch to tensorflow gpu here
  });
  /*
    Every BaseNeuralNetwork requires you to implement a function that add layers to a sequential tensorflow model
  */
  nnLR.generateLayers = function generateLayers(x_matrix, y_matrix) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    this.model.add(this.tf.layers.dense({ units: (xShape[ 1 ] * 2), inputShape: [ xShape[ 1 ], ], kernelInitializer: 'randomNormal', activation: 'relu', }));
    this.model.add(this.tf.layers.dense({ units: yShape[ 1 ], kernelInitializer: 'randomNormal', }));
  };
  console.log('training model');
  await nnLR.train(x_matrix_train, y_matrix_train);
  const estimatesPredictions = await nnLR.predict(x_matrix_test);
  /* 
  estimates = [ 
    [ 1.1274346113204956 ],
    [ -0.25014978647232056 ],
    ...
  ]
  y_matrix_test = [ 
    [ 1.322937476872205 ],
    [ -0.09055092953155416 ],
    ...
  ]
  */
  const estimatedValues = CSVDataSet.reverseColumnMatrix({ vectors: estimatesPredictions, labels: dependentVariables, });
  const actualValues = CSVDataSet.reverseColumnMatrix({ vectors: y_matrix_test, labels: dependentVariables, });
  /*
  estimatedValues = [ 
    { MEDV: 1.4987640380859375 },
    { MEDV: -0.29192864894866943 },
    ...
  ];
  actualValues = [ 
    { MEDV: 1.322937476872205 },
    { MEDV: -0.09055092953155416 },
    ...
  ];
  */
  const estimatesDescaled = estimatedValues.map(val=>CSVDataSet.inverseTransformObject(val));
  const actualsDescaled = actualValues.map(val =>CSVDataSet.inverseTransformObject(val));
  /*
  estimatesDescaled = [ 
     { MEDV: 19.4,
    ...
    LSTAT: NaN },
  { MEDV: 18.8,
    ...
    LSTAT: NaN },
    ...
  ];
  actuals = [ 
    { MEDV: 19.4,
    ...
    LSTAT: NaN },
    { MEDV: 18.8,
    ...
    LSTAT: NaN },
    ...
  ];
  */
  const estimates = ms.DataSet.columnArray('MEDV', { data: estimatesDescaled, });
  const actuals = ms.DataSet.columnArray('MEDV', { data: actualsDescaled, });
  /*
  estimates = [ 
    32.645493167885206,
    21.001051018109685,
    18.9893185314802,
  ]
  estimates = [ 
    34.7,
    21.7,
    20.4,
  ]
  */
  const accuracy = ms.util.rSquared(actuals, estimates);
  console.log({ accuracy,  }); // { accuracy: 0.8019602806119276  } ~ 82%

  const newScaledPredictions = [
    { CRIM: 0.00632, ZN: 18, INDUS: 2.31, CHAS: 0, NOX: 0.538, RM: 6.575, AGE: 65.2, DIS: 4.09, RAD: 1, TAX: 296, PTRATIO: 15.3, B: 396.9, LSTAT: 4.98, MEDV: undefined, },
    { CRIM: 0.02731, ZN: 0, INDUS: 7.07, CHAS: 0, NOX: 0.469, RM: 6.421, AGE: 78.9, DIS: 4.9671, RAD: 2, TAX: 242, PTRATIO: 17.8, B: 396.9, LSTAT: 9.14, MEDV: undefined, },
  ].map(pred=>CSVDataSet.transformObject(pred));

  /*
  newScaledPredictions = [{ 
    CRIM: -0.41936692921321594,
    ZN: 0.2845482693404666, 
    ...
    MEDV: undefined,
    },
     ...]
  inputMatrix = [ [ -0.41936692921321594,
       0.2845482693404666,
       -1.2866362317172035,
       -0.272329067679207,
       -0.1440748547324509,
       0.4132629204530747,
       -0.119894767215809,
       0.1400749839795629,
       -0.981871187861867,
       -0.6659491794887338,
       -1.457557967289609,
       0.4406158949991029,
       -1.074498970343932 ],
     [ -0.41692666996409716,
       -0.4872401872268264,
       -0.5927943782429392,
       -0.272329067679207,
       -0.7395303607434242,
       0.1940823874370036,
       0.3668034264326209,
       0.5566090495704026,
       -0.8670244885881488,
       -0.9863533804386945,
       -0.3027944997494681,
       0.4406158949991029,
       -0.49195252491856634 ] ],
  */
  const inputMatrix = CSVDataSet.columnMatrix(independentVariables, newScaledPredictions);
  const newPredictions = await nnLR.predict(inputMatrix);
  /* newPredictions= [ [ 0.49619558453559875 ], [ 0.08225022256374359 ] ] 
  */
  const newPreds = CSVDataSet.reverseColumnMatrix({ vectors: newPredictions, labels: dependentVariables, });
  /*
  newPreds= [ { MEDV: 0.43889832496643066 }, { MEDV: 0.04754585400223732 } ]  
  */
  
  const newpredsDescaled = newPreds
    .map(val => CSVDataSet.inverseTransformObject(val))
    .map(p=>({ MEDV:p.MEDV, }));
  /*
  newpredsDescaled =
   [ { MEDV: 27.467049715534355 }, { MEDV: 24.038590026697428 } ]
  */
  
}

main();