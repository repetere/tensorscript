// import ms from 'modelscript';
// import path from 'path';
// import { LSTMTimeSeries, } from '../../../lib/lstm_time_series.mjs';

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
//   const independentVariables = [
//     'Passengers',
//   ];
//   const dependentVariables = [
//     'Passengers',
//   ];
//   // const columns = independentVariables.concat(dependentVariables);
//   const csvData = await ms.csv.loadCSV('./test/mock/data/international-airline-passengers-no_footer.csv');
//   const DataSet = new ms.DataSet(csvData);
//   const scaledData = DataSet.fitColumns({
//     columns: dependentVariables.map(scaleColumnMap),
//     returnData:true,
//   });
//   const train_size = parseInt(DataSet.data.length * 0.67);
//   const test_size = DataSet.data.length - train_size;
//   // console.log({ train_size, test_size, });
//   const train_x_data = DataSet.data.slice(0, train_size);
//   // const test_x_data = DataSet.data.slice(train_size, (train_size + 6));
//   const test_x_data = DataSet.data.slice(train_size, DataSet.data.length);
//   // console.log({ test_x_data });
//   const trainDataSet = new ms.DataSet(train_x_data);
//   const testDataSet = new ms.DataSet(test_x_data);
//   const x_matrix = trainDataSet.columnMatrix(independentVariables); 
//   const x_matrix_test = testDataSet.columnMatrix(independentVariables); 
//   console.log({ x_matrix, x_matrix_test });

//   const TSTS = new LSTMTimeSeries({
//     lookBack: 3,
//   });
//   const TSTSStateful = new LSTMTimeSeries({
//     lookBack: 3,
//     stateful:true,
//   });
//   const TSTSONE = new LSTMTimeSeries({
//     lookBack: 1,
//   });
//   const TSTSStatefulONE = new LSTMTimeSeries({
//     lookBack: 1,
//     stateful:true,
//   });
//   // const m = await TSTS.train(x_matrix);
//   // const mstate = await TSTSStateful.train(x_matrix);
//   // const mone = await TSTSONE.train(x_matrix);
//   // const mstateone = await TSTSStatefulONE.train(x_matrix);
//   const input_x_one = [
//     [ 
//       DataSet.scalers.get('Passengers').scale(112),
//     ],
//     [
//       DataSet.scalers.get('Passengers').scale(118),
//     ],
//     [
//       DataSet.scalers.get('Passengers').scale(132),
//     ],
//   ];
//   const input_x = [
//     [ 
//       DataSet.scalers.get('Passengers').scale(112),
//       DataSet.scalers.get('Passengers').scale(118),
//       DataSet.scalers.get('Passengers').scale(132),
//     ], 
//     [ 
//       DataSet.scalers.get('Passengers').scale(118),
//       DataSet.scalers.get('Passengers').scale(132),
//       DataSet.scalers.get('Passengers').scale(129),
//     ], 
//     [ 
//       DataSet.scalers.get('Passengers').scale(132),
//       DataSet.scalers.get('Passengers').scale(129),
//       DataSet.scalers.get('Passengers').scale(121),
//     ], 
//     // [ 
//     //   DataSet.scalers.get('Passengers').scale(606),
//     //   DataSet.scalers.get('Passengers').scale(508),
//     //   DataSet.scalers.get('Passengers').scale(461),
//     // ], 
//     // [ 
//     //   DataSet.scalers.get('Passengers').scale(508),
//     //   DataSet.scalers.get('Passengers').scale(461),
//     //   DataSet.scalers.get('Passengers').scale(390),
//     // ], 
//   ];
//   const evals = [
//     {
//       model: TSTS,
//       modelname: 'TSTS',
//       input: input_x,
//     },
//     {
//       model: TSTSStateful,
//       modelname: 'TSTSStateful',
//       input: input_x,
//     },
//     {
//       model: TSTSONE,
//       modelname: 'TSTSONE',
//       input: input_x_one,
//     },
//     {
//       model: TSTSStatefulONE,
//       modelname: 'TSTSStatefulONE',
//       input: input_x_one,
//     },
//   ];
//   for (let i in evals) {
//     let preddata = evals[ i ];
//     const m = await preddata.model.train(x_matrix);
//     const testData = preddata.model.getTimeseriesDataSet(x_matrix_test);
//     const preInputShape = LSTMTimeSeries.getInputShape(preddata.input);
//     // console.log({ testData })
//     // console.log({preInputShape})
//     // console.log('preddata.input',preddata.input)
//     // console.log('testData.x_matrix',testData.x_matrix)
//     // const predictions = await preddata.model.predict(preddata.input);
//     const predictions = await preddata.model.predict(testData.x_matrix);
//     const predictions_unscaled = predictions.map(pred => [DataSet.scalers.get('Passengers').descale(pred[ 0 ]),]);
//     const actuals_unscaled = testData.y_matrix.map(act => [DataSet.scalers.get('Passengers').descale(act[ 0 ]),]);
//     let results = ms.DataSet.reverseColumnMatrix({
//       vectors: predictions_unscaled,
//       labels: dependentVariables,
//     });
//     console.log({
//       model: preddata.modelname,
//       // predictions,
//       // actuals_unscaled,
//       // predictions_unscaled,
//       // results,
//       accuracy: (ms.util.rSquared(
//         ms.util.pivotVector(actuals_unscaled)[ 0 ], //actuals,
//         ms.util.pivotVector(predictions_unscaled)[ 0 ], //estimates,
//       ) * 100).toFixed(2)+'%',
//     });
//   }
//   console.log('built network');
// }
// console.log({LSTMTimeSeries})
// mainTS();


// /*

// class LSTMTimeSeries extends BaseNeuralNetwork {
//   constructor(options = {}, properties) {
//     const config = Object.assign({
//       layers: [],
//       type: 'simple',
//       stateful:false,
//       stacked: false,
//       mulitpleTimeSteps:false,
//       lookBack:1,
//       batchSize:1,
//       timeSteps:1,
//       learningRate:0.1,
//       compile: {
//         loss: 'meanSquaredError',
//         optimizer: 'adam'
//       },
//       fit: {
//         epochs: 100,
//         batchSize: 1,
//       },
//     }, options);
//     super(config, properties);
//     this.createDataset = LSTMTimeSeries.createDataset;
//     this.getTimeseriesDataSet = LSTMTimeSeries.getTimeseriesDataSet;
//     this.getTimeseriesShape = LSTMTimeSeries.getTimeseriesShape;
//     return this;
//   }
//   static createDataset(dataset=[], look_back = 1) { 
//     const dataX = [];
//     const dataY = [];
//     for (let index in range(dataset.length - look_back - 1)) {
//       let i = parseInt(index);
//       let a = dataset.slice(i, i + look_back);
//       dataX.push(a);
//       dataY.push(dataset[ i + look_back ]);
//     }
//     return [dataX, dataY];
//   }
//   generateLayers(x_matrix, y_matrix, layers) {
//     const xShape = this.getInputShape(x_matrix);
//     const yShape = this.getInputShape(y_matrix);
//     this.yShape = yShape;
//     this.xShape = xShape;
//   //   const sgdoptimizer = this.tf.train.sgd(this.settings.learningRate);
//     const lstmLayers = [];
//     const rnnLayers = [];
//     const denseLayers = [];
//     if (layers) {
//       if(layers.lstmLayers && layers.lstmLayers.length) lstmLayers.push(...layers.lstmLayers);
//       if(layers.rnnLayers && layers.rnnLayers.length) rnnLayers.push(...layers.rnnLayers);
//       if(layers.denseLayers && layers.denseLayers.length) denseLayers.push(...layers.denseLayers);
//     } else if (this.settings.stateful) {
//       const batchInputShape = [ this.settings.fit.batchSize, this.settings.lookBack, 1 ];
//       console.log({batchInputShape, xShape, yShape})
//       rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  returnSequences:true, });
//       rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  });
//       denseLayers.push({ units: yShape[1], });
//     // } else if(this.settings.stacked) {
//     //   lstmLayers.push({ units: 4, inputShape: [ 1, this.settings.lookBack ], });
//     //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
//     //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
//     //   denseLayers.push({ units: yShape[1], });
//     } else {
//       const inputShape= [ 1, this.settings.lookBack ];
//       console.log('default timeseries',{inputShape, xShape,yShape})
//       lstmLayers.push({ units:4, inputShape,  });
//       denseLayers.push({ units: yShape[1], });
//     }
//     // console.log('lstmLayers',lstmLayers)
//     // console.log('denseLayers',denseLayers)
//     if (lstmLayers.length) {
//       lstmLayers.forEach(layer => {
//         this.model.add(this.tf.layers.lstm(layer));
//       });
//     }
//     if (rnnLayers.length) {
//       rnnLayers.forEach(layer => {
//         this.model.add(this.tf.layers.simpleRNN(layer));
//       });
//     }
//     if (denseLayers.length) {
//       denseLayers.forEach(layer => {
//         this.model.add(this.tf.layers.dense(layer));
//       });
//     }
//     this.layers = {
//       lstmLayers,
//       rnnLayers,
//       denseLayers,
//     };
//     // this.settings.compile.optimizer = sgdoptimizer;
//   }
//   //reshape input to be [samples, time steps, features]
//   static getTimeseriesShape(x_timeseries) {
//     const time_steps = this.settings.timeSteps;
//     const xShape = this.getInputShape(x_timeseries);
//     const _samples = xShape[ 0 ];
//     const _timeSteps = time_steps;
//     const _features = xShape[ 1 ];
//     const newShape = (this.settings.mulitpleTimeSteps || this.settings.stateful)
//       ? [ _samples,  _features, _timeSteps, ]
//       : [ _samples, _timeSteps, _features, ];
//     return newShape;
//   }
//   static getTimeseriesDataSet(timeseries, look_back) {
//     const lookBack = look_back || this.settings.lookBack;
//     const matrices = LSTMTimeSeries.createDataset.call(this, timeseries, lookBack);
//     const x_matrix = matrices[ 0 ];
//     const y_matrix = matrices[ 1 ];
//     // const timeseriesShape = LSTMTimeSeries.getTimeseriesShape.call(this,x_matrix);
//     const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, [x_matrix.length, lookBack]);
//     const xShape = BaseNeuralNetwork.getInputShape(x_matrix_timeseries);
//     const yShape = BaseNeuralNetwork.getInputShape(y_matrix);
//     return {
//       yShape,
//       xShape,
//       y_matrix,
//       x_matrix:x_matrix_timeseries,
//     }
//   }
//   async train(x_timeseries, y_timeseries, layers, x_test, y_test) {
//     let xShape;
//     let yShape;
//     let x_matrix;
//     let y_matrix;
//     const look_back = this.settings.lookBack;
//     const time_steps = this.settings.timeSteps;
//     if (y_timeseries) {
//       x_matrix = x_timeseries;
//       y_matrix = y_timeseries;
//     } else {
//       const matrices = this.createDataset(x_timeseries, look_back);
//       // console.log({matrices})
//       x_matrix = matrices[ 0 ];
//       y_matrix = matrices[ 1 ];
//       xShape = this.getInputShape(x_matrix);
//       yShape = this.getInputShape(y_matrix);
//     }
//     //_samples, _timeSteps, _features
//     const timeseriesShape = this.getTimeseriesShape(x_matrix);
//     const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
//     // console.log({
//     //   x_matrix_timeseries,timeseriesShape, yShape,
//     //   // y_matrix, x_matrix
//     // });
//     // console.log('x_matrix_timeseries',JSON.stringify(x_matrix_timeseries));
//     // console.log('x_matrix',JSON.stringify(x_matrix));
//     // console.log('y_matrix',JSON.stringify(y_matrix));
//     const xs = this.tf.tensor(x_matrix_timeseries, timeseriesShape);
//     const ys = this.tf.tensor(y_matrix, yShape);
//     this.xShape = timeseriesShape;
//     this.yShape = yShape;
//     this.model = this.tf.sequential();
//     this.generateLayers.call(this, x_matrix_timeseries, y_matrix, layers || this.layers, x_test, y_test);
//     this.model.compile(this.settings.compile);
//     if (this.settings.stateful) {
//       this.settings.fit.shuffle = false;
//     }
//     await this.model.fit(xs, ys, this.settings.fit);
//     this.model.summary()
//     xs.dispose();
//     ys.dispose();
//     return this.model;
//   }
//   calculate(x_matrix) {
//     const timeseriesShape = this.getTimeseriesShape(x_matrix);
//     const input_matrix = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
//     const x_matrix_timeseriesShape = this.getInputShape(input_matrix);
//     // console.log({ timeseriesShape, input_matrix, x_matrix_timeseriesShape, });
//     return super.calculate(input_matrix);
//   }
//   async predict(input_matrix, options) {
//     if (this.settings.stateful && input_matrix.length > 1) {
//       return Promise.all(input_matrix.map(input=>super.predict([input], options))) ;
//     } else {
//       return super.predict(input_matrix, options);
//     }
//   }
// }
// */