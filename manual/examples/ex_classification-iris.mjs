/**
 * Node 8+
 * $ node --experimental-modules manual/examples/ex_classification-iris.mjs
 */
import ms from 'modelscript'; // used for scaling, data manipulation
import { DeepLearningClassification, } from '../../index.mjs';
import { scaleColumnMap, } from './helper';
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
    columns: [
      {
        name: 'plant',
        options: {
          strategy: 'onehot',
        },
      },
    ],
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
    epochs: 100,
    batchSize:10,
  };
  const nnClassification = new DeepLearningClassification({ fit, });
  await nnClassification.train(x_matrix_train, y_matrix_train);
  const estimates = await nnClassification.predict(x_matrix_test, { probability: false, });
  /* 
  estimates = [ 
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    [ 1, 0, 0 ],
    ...
  ]
  */
  const estimatedValues = ms.DataSet.reverseColumnMatrix({ vectors: estimates, labels: dependentVariables, });
  // console.log('estimatedValues', estimatedValues);
  // console.log('y_matrix_test', y_matrix_test);
  // const CM = ConfusionMatrix.fromLabels([[1],[2],[3]], [[1],[2],[3]]);
  // const accuracy = CM.getAccuracy();
}

main();