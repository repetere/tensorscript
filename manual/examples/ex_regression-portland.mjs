/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_regression-boston.mjs
 */
import * as ms from 'modelscript'; // used for scaling, data manipulation
import { MultipleLinearRegression, } from '../../index.mjs';
//if running on node
// import tf from '@tensorflow/tfjs-node';


const independentVariables = ['sqft', 'bedrooms', ];
const dependentVariables = ['price',];

async function main() {
  /*
  CSVData = [
    { sqft: 2104, bedrooms: 3, price: 399900 },
    { sqft: 1600, bedrooms: 3, price: 329900 },
    ...
    { sqft: 1203, bedrooms: 3, price: 239500 } 
  ]
  */
  const CSVData = await ms.csv.loadCSV('./test/mock/data/portland_housing_data.csv');
  const CSVDataSet = new ms.DataSet(CSVData);
  CSVDataSet.fitColumns({
    sqft: ['scale', 'standard',],
    bedrooms: ['scale', 'standard',],
    price: ['scale', 'standard',],
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
    [2014, 3],
    [1600, 3],
    ...
  ]
  y_matrix_test = [ 
    [399900],
    [329900],
    ...
  ]
  */
  const fit = {
    epochs: 100,
    batchSize:5,
  };
  const nnLR = new MultipleLinearRegression({ fit, }, {
    // tf - can switch to tensorflow gpu here
  });
  console.log('training model');
  await nnLR.train(x_matrix_train, y_matrix_train);
  const estimatesPredictions = await nnLR.predict(x_matrix_test);
  /* 
  estimatesPredictions = [ 
    [ -0.09963418543338776 ],
    [ 2.1494181156158447 ],
    ...
  ]
  y_matrix_test = [ 
    [ -0.0832826930356929 ],
    [ 2.874981038969331 ],
    ...
  ]
  */
  const estimatedValues = CSVDataSet.reverseColumnMatrix({ vectors: estimatesPredictions, labels: dependentVariables, });
  const actualValues = CSVDataSet.reverseColumnMatrix({ vectors: y_matrix_test, labels: dependentVariables, });
  /*
  estimatedValues = [ 
    { price: -0.0832826930356929 },
    { price: 2.874981038969331 },
    ...
  ];
  actualValues = [ 
    { price: -0.097346231341362 },
    { price: 2.206512451171875 },
    ...
  ];
  */
  const estimatesDescaled = estimatedValues.map(val=>CSVDataSet.inverseTransformObject(val));
  const actualsDescaled = actualValues.map(val =>CSVDataSet.inverseTransformObject(val));
  /*
  estimatesDescaled = [ 
    { price: 328581.1387223731,
    ...
    bedrooms:: NaN },
    { price: 610483.8580602541,
    ...
    bedrooms:: NaN },
    ...
  ];
  actuals = [ 
    { price: 329999,,
    ...
    bedrooms:: NaN },
    { price: 699900,,
    ...
    bedrooms:: NaN },
    ...
  ];
  */

  const estimates = ms.DataSet.columnArray('price', { data: estimatesDescaled, });
  const actuals = ms.DataSet.columnArray('price', { data: actualsDescaled, });
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
  console.log({ accuracy,  }); // { accuracy: 0.8596126619239105  } ~ 86%

  const newScaledPredictions = [
    { sqft: 4215, bedrooms: 4, price: undefined, },
    { sqft: 852, bedrooms: 2, price: undefined, },
  ].map(pred => CSVDataSet.transformObject(pred));

  /*
  newScaledPredictions = [
    { 
      sqft: 2.78635030976002,
      bedrooms: 1.0904165374468842,
      price: NaN 
    },
    { 
      sqft: -1.4454227371491544,
      bedrooms: -1.5377669117840669,
      price: NaN 
    }
  ]
  inputMatrix = [ 
    [ 2.78635030976002, 1.0904165374468842 ],
    [ -1.4454227371491544, -1.5377669117840669 ] 
  ],
  */
  const inputMatrix = CSVDataSet.columnMatrix(independentVariables, newScaledPredictions);
  const newPredictions = await nnLR.predict(inputMatrix);
  /* newPredictions= [ [ 2.135737180709839 ], [ -0.9959254860877991 ] ] */
  const newPreds = CSVDataSet.reverseColumnMatrix({ vectors: newPredictions, labels: dependentVariables, });
  /* newPreds= [ { price: 2.135737180709839 }, { price: -0.9959254860877991 } ] ] */

  const newpredsDescaled = newPreds
    .map(val => CSVDataSet.inverseTransformObject(val))
    .map(p=>({ price:p.price, }));
  /* newpredsDescaled = [ { price: 606373.7003425548 }, { price: 220151.62698092213 }  ] */
  
}

main();