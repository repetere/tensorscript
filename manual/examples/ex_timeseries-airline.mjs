/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_regression-boston.mjs
 */
import * as ms from 'modelscript'; // used for scaling, data manipulation
import { LSTMTimeSeries, } from '../../index.mjs';
//if running on node
// import tf from '@tensorflow/tfjs-node';


const independentVariables = ['Passengers',];
const dependentVariables = ['Passengers',];

async function main() {
  /*
  CSVData = [
    { Month: '1949-01', Passengers: 112 },
    { Month: '1949-02', Passengers: 118 },
    { Month: '1949-03', Passengers: 132 },
    { Month: '1949-04', Passengers: 129 },
    ...
  ]
  after scaling = [ 
    { Month: '1949-01', Passengers: -1.4028822039369186 },
    { Month: '1949-02', Passengers: -1.3528681653893018 },
    ...
  ]
  */
  const CSVData = await ms.csv.loadCSV('./test/mock/data/airline-sales.csv');
  const CSVDataSet = new ms.DataSet(CSVData);
  CSVDataSet.fitColumns({
    Passengers: ['scale', 'standard', ],
  });

  const train_size = parseInt(CSVDataSet.data.length * 0.67);
  // const test_size = CSVDataSet.data.length - train_size;
  const trainDataSet = new ms.DataSet(CSVDataSet.data.slice(0, train_size));
  const testDataSet = new ms.DataSet(CSVDataSet.data.slice(train_size, CSVDataSet.data.length));
  const x_matrix_train = trainDataSet.columnMatrix(independentVariables);
  const y_matrix_train = trainDataSet.columnMatrix(dependentVariables);
  const x_matrix_test = testDataSet.columnMatrix(independentVariables); 
  const y_matrix_test = testDataSet.columnMatrix(dependentVariables);
  /* 
  x_train = [
    [ -1.4028822039369186 ],
    [ -1.3528681653893018 ],
    ...
  ]
  y_matrix_test = [ 
    [ 0.2892594335907897 ],
    [ 0.17256001031301674 ],
    ...
  ]
  */
  const fit = {
    epochs: 200,
    batchSize:5,
  };
  const lstmTS = new LSTMTimeSeries({ fit, }, {
    // tf - can switch to tensorflow gpu here
  });
  console.log('training model');
  await lstmTS.train(x_matrix_train);
  const estimatesPredictions = await lstmTS.predict(x_matrix_test);
  /* 
  estimatesPredictions = [ 
    [ 0.3164129853248596 ],
    [ 0.22678324580192566 ],
    ...
  ]
  y_matrix_test = [ 
    [ 0.2892594335907897 ],
    [ 0.17256001031301674 ],
    ...
  ]
  */
  const estimatedValues = CSVDataSet.reverseColumnMatrix({ vectors: estimatesPredictions, labels: dependentVariables, });
  const actualValues = CSVDataSet.reverseColumnMatrix({ vectors: y_matrix_test, labels: dependentVariables, });
  /*
  estimatedValues = [ 
    { Passengers: 0.29080596566200256 },
    { Passengers: 0.20487788319587708 },
    ...
  ];
  actualValues = [ 
    { Passengers: 0.2892594335907897 },
    { Passengers: 0.17256001031301674 },
    ...
  ];
  */
  const estimatesDescaled = estimatedValues.map(val=>CSVDataSet.inverseTransformObject(val));
  const actualsDescaled = actualValues.map(val => CSVDataSet.inverseTransformObject(val));

  /*
  estimatesDescaled = [ 
    { Passengers: 315.18553175661754 },
    { Passengers: 304.87705618118696 },
    ...
  ];
  actuals = [ 
    { Passengers: 315 },
    { Passengers: 301 },
    ...
  ];
  */

  const estimates = ms.DataSet.columnArray('Passengers', { data: estimatesDescaled, });
  const actuals = ms.DataSet.columnArray('Passengers', { data: actualsDescaled, });
  /*
  estimates = [ 
    328772.6900112897,
    620789.3324962095,
  ]
  actuals = [ 
    329999,
    699900,
  ]
  */

  const accuracy = ms.util.rSquared(actuals, estimates);
  console.log({ accuracy,  }); // { accuracy: 0.0.768662218946419  } ~ 77%

  const newScaledPredictions = [
    { Month: '1960-12', Passengers: estimates[estimates.length -1], },
  ].map(pred => CSVDataSet.transformObject(pred));

  /*
  newScaledPredictions = [
    { Month: '1960-12', Passengers: 0.8113135695457461 }
  ]
  inputMatrix = [ 
    [ 0.7591229677200318 ]
  ],
  */
  const inputMatrix = CSVDataSet.columnMatrix(independentVariables, newScaledPredictions);
  const newPredictions = await lstmTS.predict(inputMatrix);
  /* newPredictions= [  [ 0.6041761636734009 ]  ] */
  const newPreds = CSVDataSet.reverseColumnMatrix({ vectors: newPredictions, labels: dependentVariables, });
  /* newPreds= [ { Passengers: 0.6041761636734009 } ] ] */

  const newpredsDescaled = newPreds
    .map(val => CSVDataSet.inverseTransformObject(val))
    .map(p=>({ Passengers:p.Passengers, }));
  /* newpredsDescaled = [ { Passengers: 352.77940025172586 }  ] */
  
}

main();