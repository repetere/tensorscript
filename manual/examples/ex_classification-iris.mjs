/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_classification-iris.mjs
 */
import * as ms from 'modelscript'; // used for scaling, data manipulation
import { DeepLearningClassification, } from '../../index.mjs';
//if running on node
// import tf from '@tensorflow/tfjs-node';
const ConfusionMatrix = ms.ml.ConfusionMatrix;

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

async function main() {
  const CSVData = await ms.csv.loadCSV('./test/mock/data/iris_data.csv');
  const DataSet = new ms.DataSet(CSVData);
  DataSet.fitColumns({
    plant: 'onehot',
  });
  const testTrainSplit = ms.cross_validation.train_test_split(DataSet.data, { train_size: 0.7, });
  const { train, test, } = testTrainSplit;
  const testDataSet = new ms.DataSet(test);
  const trainDataSet = new ms.DataSet(train);
  const x_matrix_train = trainDataSet.columnMatrix(independentVariables);
  const y_matrix_train = trainDataSet.columnMatrix(dependentVariables);
  const x_matrix_test = testDataSet.columnMatrix(independentVariables); 
  const y_matrix_test = testDataSet.columnMatrix(dependentVariables);
  /* 
  y_matrix_test = [ 
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    ...
  ]
  */
  const fit = {
    epochs: 200,
    batchSize:10,
  };
  const nnClassification = new DeepLearningClassification({ fit, }, {
    // tf - can switch to tensorflow gpu here
  });
  console.log('training model');
  await nnClassification.train(x_matrix_train, y_matrix_train);
  const estimatesPredictions = await nnClassification.predict(x_matrix_test, { probability: false, });
  /* 
  estimates = [ 
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    ...
  ]
  y_matrix_test = [ 
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    ...
  ]
  */
  const estimatedValues = ms.DataSet.reverseColumnMatrix({ vectors: estimatesPredictions, labels: dependentVariables, });
  const actualValues = ms.DataSet.reverseColumnMatrix({ vectors: y_matrix_test, labels: dependentVariables, });
  /*
  estimatedValues = [ 
    { 'plant_Iris-setosa': 1, 'plant_Iris-versicolor': 0, 'plant_Iris-virginica': 0 },
    { 'plant_Iris-setosa': 1, 'plant_Iris-versicolor': 0, 'plant_Iris-virginica': 0 },
    ...
  ];
  actualValues = [ 
    { 'plant_Iris-setosa': 1, 'plant_Iris-versicolor': 0, 'plant_Iris-virginica': 0 },
    { 'plant_Iris-setosa': 1, 'plant_Iris-versicolor': 0, 'plant_Iris-virginica': 0 },
    ...
  ];
  */
  const reformattedEstimatesValues = DataSet.oneHotDecoder('plant', { data: estimatedValues, });
  const reformattedActualValues = DataSet.oneHotDecoder('plant', { data: actualValues, });
  /*
  reformattedEstimatesValues = [ 
    { plant: 'Iris-setosa' },
    { plant: 'Iris-setosa' },
    ...
  ];
  reformattedActualValues = [ 
    { plant: 'Iris-setosa' },
    { plant: 'Iris-setosa' },
    ...
  ];
  */
  const estimates = ms.DataSet.columnArray('plant', { data: reformattedEstimatesValues, });
  const actuals = ms.DataSet.columnArray('plant', { data: reformattedActualValues, });
  /*
  estimates = [ 
    'Iris-setosa',
    'Iris-setosa',
    ...
  ];
  actuals = [ 
    'Iris-setosa',
    'Iris-setosa',
    ...
  ];
  */
  const CM = ConfusionMatrix.fromLabels(actuals,estimates);
  const accuracy = CM.getAccuracy();
  console.log({ accuracy, }); // { accuracy: 0.9111111111111111 } ~ 91%
}

main();