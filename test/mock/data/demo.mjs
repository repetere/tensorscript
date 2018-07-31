// import ms from 'modelscript';
// import { LSTMMultivariateTimeSeries, } from '../../../lib/lstm_multivariate_time_series.mjs';
// import range from 'lodash.range';

// const independentVariables = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', ];
// const dependentVariables = ['o1',];

// const ds = [
//   [10, 20, 30, 40, 50, 60, 70, 80, 90,],
//   [11, 21, 31, 41, 51, 61, 71, 81, 91,],
//   [12, 22, 32, 42, 52, 62, 72, 82, 92,],
//   [13, 23, 33, 43, 53, 63, 73, 83, 93,],
//   [14, 24, 34, 44, 54, 64, 74, 84, 94,],
//   [15, 25, 35, 45, 55, 65, 75, 85, 95,],
//   [16, 26, 36, 46, 56, 66, 76, 86, 96,],
//   [17, 27, 37, 47, 57, 67, 77, 87, 97,],
//   [18, 28, 38, 48, 58, 68, 78, 88, 98,],
//   [19, 29, 39, 49, 59, 69, 79, 89, 99,],
// ];


// function seriesToSupervised(data, n_in = 1, n_out = 1) {
//   // let n_vars = data[ 0 ].length;
//   let cols = [];
//   // let names = [];
//   // input sequence (t-n, ... t-1)
//   for (let x in data) {
//     x = parseInt(x);
//     let maxIndex = x + n_in + n_out;
//     if (maxIndex > data.length) break;
//     cols[ x ] = [];
//     // console.log({ x,maxIndex });
//     for (let i in range(n_in)) {
//       i = parseInt(i);
//       cols[ x ].push(...data[x+i]);
//       // console.log({ i, cols, });
//     }
//     for (let j in range(n_out)){
//       j = parseInt(j);
//       cols[ x ].push(...data[ x+j+n_in ]);
//       // console.log({ j, cols, });
//     }
//   }
//   return cols;
// }
// function drop(data,columns) {
//   return data.reduce((cols, row, i) => { 
//     cols[ i ] = [];
//     row.forEach((col, idx) => {
//       if (columns.indexOf(idx)===-1) {
//         cols[ i ].push(col);
//       }
//     });
//     return cols;
//   }, []);
// }

// function getDropableColumns(features, n_in, n_out) {
//   const cols = features + 1;
//   const total_cols = cols * n_in + cols * n_out;
//   // console.log({ cols, total_cols });
//   return range(total_cols - cols +1, total_cols);
// }

// // const n_in = 1; //lookbacks
// // const n_out = 1;
// // const series = seriesToSupervised(ds, n_in, n_out);
// // const dropped = getDropableColumns(8, n_in, n_out);
// // // const droppedColumns = drop(series, [ 10, 11, 12, 13, 14, 15, 16, 17, ]);
// // const droppedColumns = drop(series, dropped);
// // const y = ms.util.pivotVector(droppedColumns)[ droppedColumns[0].length - 1 ];
// // const x = drop(droppedColumns, [ droppedColumns[ 0 ].length - 1 ]);
// // console.log({ series, dropped, droppedColumns, y, x, });


// function scaleColumnMap(columnName) {
//   return {
//     name: columnName,
//     options: {
//       strategy: 'scale',
//       scaleOptions: {
//         strategy:'standard',
//       },
//     },
//   };
// }



// async function mainTS() {
//   const csvData = await ms.csv.loadCSV('./test/mock/data/sample.csv');
//   const DataSet = new ms.DataSet(csvData);
//   const columns = dependentVariables.concat(independentVariables);
//   const scaledData = DataSet.fitColumns({
//     columns: columns.map(scaleColumnMap),
//     returnData: true,
//   });
  
//   const train_size = parseInt(DataSet.data.length * 0.67);
//   const test_size = DataSet.data.length - train_size;
//   // console.log({ train_size, test_size });
  
//   const train_x_data = DataSet.data.slice(0, train_size);
//   const test_x_data = DataSet.data.slice(train_size, DataSet.data.length);
//   const trainDataSet = new ms.DataSet(train_x_data);
//   const testDataSet = new ms.DataSet(test_x_data);
//   const x_matrix = trainDataSet.columnMatrix(columns); 
//   const x_matrix_test = testDataSet.columnMatrix(columns); 
//   // console.log({ csvData, scaledData, });

//   const TSTSONE = new LSTMMultivariateTimeSeries({
//     lookback: 1,
//     features: 8,
//   });
//   const TSTS = new LSTMMultivariateTimeSeries({
//     lookback: 2,
//     features: 8,
//   });
//   const evals = [
//     // {
//     //   model: TSTS,
//     //   modelname: 'TSTS',
//     // },
//     {
//       model: TSTSONE,
//       modelname: 'TSTSONE',
//     },
//   ];
//   for (let i in evals) {
//     let preddata = evals[ i ];
//     const testData = preddata.model.getTimeseriesDataSet(x_matrix_test);
//     const m = await preddata.model.train(x_matrix,undefined,undefined,testData.x_matrix,testData.y_matrix);
//     // const preInputShape = LSTMTimeSeries.getInputShape(preddata.input);
//     // // const predictions = await preddata.model.predict(preddata.input);
//     // const predictions = await preddata.model.predict(x_matrix_test);
//     const predictions = await preddata.model.predict(testData.x_matrix);
//     const predictions_unscaled = predictions.map(pred => [DataSet.scalers.get('o1').descale(pred[ 0 ]),]);
//     const actuals_unscaled = testData.y_matrix.map(act => [DataSet.scalers.get('o1').descale(act[ 0 ]),]);
//     // let results = ms.DataSet.reverseColumnMatrix({
//     //   vectors: predictions_unscaled,
//     //   labels: dependentVariables,
//     // });
//     console.log({
//       //   model: preddata.modelname,
//       predictions,
//       actuals_unscaled,
//       predictions_unscaled,
//       //   // results,
//       accuracy: (ms.util.rSquared(
//         ms.util.pivotVector(actuals_unscaled)[ 0 ], //actuals,
//         ms.util.pivotVector(predictions_unscaled)[ 0 ], //estimates,
//       ) * 100).toFixed(2)+'%',
//     });
//   }
//   console.log('built network');
// }
// mainTS();