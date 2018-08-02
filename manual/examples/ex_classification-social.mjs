/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_classification-social.mjs
 */
import * as ms from 'modelscript'; // used for scaling, data manipulation
import { LogisticRegression, } from '../../index.mjs';
//if running on node
// import tf from '@tensorflow/tfjs-node';
const ConfusionMatrix = ms.ml.ConfusionMatrix;

const independentVariables = [
  'Age',
  'EstimatedSalary',
];
const dependentVariables = [
  'Purchased',
];

async function main() {
  /*
  CSVData = [
    { 
      'User ID': 15709441,
      Gender: 'Female',
      Age: 35,
      EstimatedSalary: 44000,
      Purchased: 0 
    },
    { 
      'User ID': 15710257,
      Gender: 'Female',
      Age: 35,
      EstimatedSalary: 25000,
      Purchased: 0 
    },
    ...
  ]
  */
  const CSVData = await ms.csv.loadCSV('./test/mock/data/social_network_ads.csv');
  const CSVDataSet = new ms.DataSet(CSVData);
  CSVDataSet.fitColumns({
    Age: ['scale', 'standard',],
    EstimatedSalary: ['scale', 'standard',],
  });
  const testTrainSplit = ms.cross_validation.train_test_split(CSVDataSet.data, { train_size: 0.7, });
  const { train, test, } = testTrainSplit;
  const testDataSet = new ms.DataSet(test);
  const trainDataSet = new ms.DataSet(train);
  const x_matrix_train = trainDataSet.columnMatrix(independentVariables);
  const y_matrix_train = trainDataSet.columnMatrix(dependentVariables);
  const x_matrix_test = testDataSet.columnMatrix(independentVariables); 
  const y_matrix_test = testDataSet.columnMatrix(dependentVariables);
  /* 
  train = [
    { 
      'User ID': 15715541,
      Gender: 'Female',
      Age: -0.9210258186650291,
      EstimatedSalary: 0.4181457784478519,
      Purchased: 0
    },
    { 
      'User ID': 15622478,
      Gender: 'Male',
      Age: 0.8914537830579694,
      EstimatedSalary: -0.7843074508252975,
      Purchased: 0 
    },
    ...
  ]
  y_matrix_test = [ 
    [ 0 ],
    [ 0 ],
    [ 0 ],
    [ 1 ],
    ...
  ]
  */
  const fit = {
    epochs: 200,
    batchSize:10,
  };
  const nnLR = new LogisticRegression({ fit, }, {
    // tf - can switch to tensorflow gpu here
  });
  console.log('training model');
  await nnLR.train(x_matrix_train, y_matrix_train);
  const estimatesPredictions = await nnLR.predict(x_matrix_test, { probability: false, });
  /* 
  estimates = [ 
    [ 0 ],
    [ 0 ],
    ...
  ]
  y_matrix_test = [ 
    [ 0 ],
    [ 0 ],
    ...
  ]
  */
  const estimatedValues = CSVDataSet.reverseColumnMatrix({ vectors: estimatesPredictions, labels: dependentVariables, });
  const actualValues = CSVDataSet.reverseColumnMatrix({ vectors: y_matrix_test, labels: dependentVariables, });
  /*
  estimatedValues = [ 
    { Purchased: 0 },
    { Purchased: 0 },
    ...
  ];
  actualValues = [ 
    { Purchased: 0 },
    { Purchased: 0 },
    ...
  ];
  */
  const estimates = CSVDataSet.columnArray('Purchased', { data: estimatedValues, });
  const actuals = CSVDataSet.columnArray('Purchased', { data: actualValues, });
  /*
  estimates = [ 
    0,
    0,
    ...
  ];
  actuals = [ 
    0,
    0,
    ...
  ];
  */
  const CM = ConfusionMatrix.fromLabels(actuals, estimates);
  const accuracy = CM.getAccuracy();
  console.log({ accuracy, }); // { accuracy: 0.8166666666666667 } ~ 82%

  const newScaledPrediction = CSVDataSet.transformObject({
    Age: 35,
    EstimatedSalary: 44000,
    'User ID': undefined,
    Gender: undefined,
    Purchased: undefined,
  });
  const newScaledPrediction2 = CSVDataSet.transformObject({
    Age: 18,
    EstimatedSalary: 32000,
    'User ID': undefined,
    Gender: undefined,
    Purchased: undefined,
  });
  const newScaledPrediction3 = CSVDataSet.transformObject({
    Age: 39,
    EstimatedSalary: 127000,
    'User ID': undefined,
    Gender: undefined,
    Purchased: undefined,
  });
  const predictionData = [ newScaledPrediction, newScaledPrediction2, newScaledPrediction3, ];
  /*
  newScaledPrediction = { Age: -0.253270175924977, EstimatedSalary: -0.75497932328205, 'User ID': undefined, Gender: undefined, Purchased: undefined } 
  inputMatrix = [ [ -0.253270175924977, -0.75497932328205 ] ]
  */
  const inputMatrix = CSVDataSet.columnMatrix(independentVariables, predictionData);
  const newPredictions = await nnLR.predict(inputMatrix, { probability: false, });
  /* newPredictions= [ 
    [ 0 ], - no (Age 35/EstimatedSalary 44000)
    [ 0 ], - no (Age 18/EstimatedSalary 32000)
    [ 1 ], - yes (Age 39/EstimatedSalary 127000)
  ]
  */

}

main();