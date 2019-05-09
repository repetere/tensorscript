import { LSTMTimeSeries, } from './lstm_time_series';
import range from 'lodash.range';

/**
 * Long Short Term Memory Multivariate Time Series with Tensorflow
 * @class LSTMMultivariateTimeSeries
 * @extends {LSTMTimeSeries}
 */
export class LSTMMultivariateTimeSeries extends LSTMTimeSeries {
  /**
   * Creates dataset data
   * @example
   * const ds = [
  [10, 20, 30, 40, 50, 60, 70, 80, 90,],
  [11, 21, 31, 41, 51, 61, 71, 81, 91,],
  [12, 22, 32, 42, 52, 62, 72, 82, 92,],
  [13, 23, 33, 43, 53, 63, 73, 83, 93,],
  [14, 24, 34, 44, 54, 64, 74, 84, 94,],
  [15, 25, 35, 45, 55, 65, 75, 85, 95,],
  [16, 26, 36, 46, 56, 66, 76, 86, 96,],
  [17, 27, 37, 47, 57, 67, 77, 87, 97,],
  [18, 28, 38, 48, 58, 68, 78, 88, 98,],
  [19, 29, 39, 49, 59, 69, 79, 89, 99,],
];
   * LSTMMultivariateTimeSeries.createDataset(ds,1) // => 
      //  [ 
      //   [ 
      //    [ 20, 30, 40, 50, 60, 70, 80, 90 ],
      //    [ 21, 31, 41, 51, 61, 71, 81, 91 ],
      //    [ 22, 32, 42, 52, 62, 72, 82, 92 ],
      //    [ 23, 33, 43, 53, 63, 73, 83, 93 ],
      //    [ 24, 34, 44, 54, 64, 74, 84, 94 ],
      //    [ 25, 35, 45, 55, 65, 75, 85, 95 ],
      //    [ 26, 36, 46, 56, 66, 76, 86, 96 ],
      //    [ 27, 37, 47, 57, 67, 77, 87, 97 ],
      //    [ 28, 38, 48, 58, 68, 78, 88, 98 ]    
      //   ], //x_matrix
      //   [ 11, 12, 13, 14, 15, 16, 17, 18, 19 ], //y_matrix
      //   8 //features
      // ]
   * @param {Array<Array<number>} dataset - array of values
   * @param {Number} look_back - number of values in each feature 
   * @override 
   * @return {[Array<Array<number>>,Array<number>]} returns x matrix and y matrix for model trainning
   */
  /* istanbul ignore next */
  static createDataset(dataset = [], look_back = 1) { 
    const features = (this.settings && this.settings.features) ? this.settings.features : dataset[ 0 ].length - 1;
    const n_in = look_back || this.settings.lookback; //1; //lookbacks
    const n_out = (this.settings && this.settings.outputs) ? this.settings.outputs : 1; //1;
    const series = LSTMMultivariateTimeSeries.seriesToSupervised(dataset, n_in, n_out);
    const dropped = LSTMMultivariateTimeSeries.getDropableColumns(features, n_in, n_out);
    const droppedColumns = LSTMMultivariateTimeSeries.drop(series, dropped);
    const dropped_c_columns = [ 0, ];
    const original_dropped_c_columns = [ 0, droppedColumns[ 0 ].length - 1, ];
    for (let i=0; i < n_in; i++){
      dropped_c_columns.push((i + 1) * features+1);
    }
    
    // console.log({ series, dropped_c_columns,original_dropped_c_columns, dropped, });
    const y = pivotVector(droppedColumns)[ droppedColumns[0].length - 1 ];
    const x = LSTMMultivariateTimeSeries.drop(
      droppedColumns,
      original_dropped_c_columns,
      // [ 0, droppedColumns[ 0 ].length - 1, ]
    );
    return [ x, y, features, ];
  }
  /**
   * Drops columns by array index
   * @example
const data = [ [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31, 41, 51, 61, 71, 81, 91 ],
     [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12, 22, 32, 42, 52, 62, 72, 82, 92 ],
     [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13, 23, 33, 43, 53, 63, 73, 83, 93 ],
     [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14, 24, 34, 44, 54, 64, 74, 84, 94 ],
     [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15, 25, 35, 45, 55, 65, 75, 85, 95 ],
     [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16, 26, 36, 46, 56, 66, 76, 86, 96 ],
     [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17, 27, 37, 47, 57, 67, 77, 87, 97 ],
     [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18, 28, 38, 48, 58, 68, 78, 88, 98 ],
     [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19, 29, 39, 49, 59, 69, 79, 89, 99 ] ];
const n_in = 1; //lookbacks
const n_out = 1;
const dropColumns = getDropableColumns(8, n_in, n_out); // =>[ 10, 11, 12, 13, 14, 15, 16, 17 ]
const newdata = drop(data,dropColumns); //=> [ 
    // [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11 ],
    // [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12 ],
    // [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13 ],
    // [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14 ],
    // [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15 ],
    // [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16 ],
    // [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17 ],
    // [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18 ],
    // [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19 ] 
    //]
  * @param {Array<Array<number>>} data - data set to drop columns 
  * @param {Array<number>} columns - array of column indexes
   * @returns {Array<Array<number>>} matrix with dropped columns
   */
  static drop(data, columns) {
    return data.reduce((cols, row, i) => { 
      cols[ i ] = [];
      row.forEach((col, idx) => {
        if (columns.indexOf(idx)===-1) {
          cols[ i ].push(col);
        }
      });
      return cols;
    }, []);
  }
  /**
   * Converts data set to supervised labels for forecasting, the first column must be the dependent variable
   * @example 
   const ds = [
    [10, 20, 30, 40, 50, 60, 70, 80, 90,],
    [11, 21, 31, 41, 51, 61, 71, 81, 91,],
    [12, 22, 32, 42, 52, 62, 72, 82, 92,],
    [13, 23, 33, 43, 53, 63, 73, 83, 93,],
    [14, 24, 34, 44, 54, 64, 74, 84, 94,],
    [15, 25, 35, 45, 55, 65, 75, 85, 95,],
    [16, 26, 36, 46, 56, 66, 76, 86, 96,],
    [17, 27, 37, 47, 57, 67, 77, 87, 97,],
    [18, 28, 38, 48, 58, 68, 78, 88, 98,],
    [19, 29, 39, 49, 59, 69, 79, 89, 99,],
  ]; 
  const n_in = 1; //lookbacks
  const n_out = 1;
  const series = seriesToSupervised(ds, n_in, n_out); //=> [ 
    // [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31, 41, 51, 61, 71, 81, 91 ],
    // [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12, 22, 32, 42, 52, 62, 72, 82, 92 ],
    // [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13, 23, 33, 43, 53, 63, 73, 83, 93 ],
    // [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14, 24, 34, 44, 54, 64, 74, 84, 94 ],
    // [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15, 25, 35, 45, 55, 65, 75, 85, 95 ],
    // [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16, 26, 36, 46, 56, 66, 76, 86, 96 ],
    // [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17, 27, 37, 47, 57, 67, 77, 87, 97 ],
    // [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18, 28, 38, 48, 58, 68, 78, 88, 98 ],
    // [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19, 29, 39, 49, 59, 69, 79, 89, 99 ] 
    //];
   * 
   * @param {Array<Array<number>>} data - data set 
   * @param {number} n_in - look backs 
   * @param {number} n_out - future iterations (only 1 supported) 
   * @todo support multiple future iterations
   * @returns {Array<Array<number>>} multivariate dataset for time series
   */
  static seriesToSupervised(data, n_in = 1, n_out = 1) {
    if (n_out !== 1) throw new RangeError('Only 1 future iteration supported');
    if (data && Array.isArray(data) && Array.isArray(data[0]) && data[ 0 ].length < 2) throw new RangeError('Must include at least two columns, the first column must be the dependent variable, followed by independent variables');
    // let n_vars = data[ 0 ].length;
    let cols = [];
    // let names = [];
    // input sequence (t-n, ... t-1)
    for (let x in data) {
      x = parseInt(x);
      let maxIndex = x + n_in + n_out;
      if (maxIndex > data.length) break;
      cols[ x ] = [];
      // console.log({ x,maxIndex });
      for (let i in range(n_in)) {
        i = parseInt(i);
        cols[ x ].push(...data[x+i]);
        // console.log({ i, cols, });
      }
      for (let j in range(n_out)){
        j = parseInt(j);
        cols[ x ].push(...data[ x+j+n_in ]);
        // console.log({ j, cols, });
      }
    }
    return cols;
  }
  /**
   * Calculates which columns to drop by index
   * @todo support multiple iterations in the future, also only one output variable supported in column features * lookbacks -1
   * @example
const ds = [
  [10, 20, 30, 40, 50, 60, 70, 80, 90,],
  [11, 21, 31, 41, 51, 61, 71, 81, 91,],
  [12, 22, 32, 42, 52, 62, 72, 82, 92,],
  [13, 23, 33, 43, 53, 63, 73, 83, 93,],
  [14, 24, 34, 44, 54, 64, 74, 84, 94,],
  [15, 25, 35, 45, 55, 65, 75, 85, 95,],
  [16, 26, 36, 46, 56, 66, 76, 86, 96,],
  [17, 27, 37, 47, 57, 67, 77, 87, 97,],
  [18, 28, 38, 48, 58, 68, 78, 88, 98,],
  [19, 29, 39, 49, 59, 69, 79, 89, 99,],
];
const n_in = 1; //lookbacks
const n_out = 1;
const dropped = getDropableColumns(8, n_in, n_out); //=> [ 10, 11, 12, 13, 14, 15, 16, 17 ]
   * @param {number} features - number of independent variables
   * @param {number} n_in - look backs 
   * @param {number} n_out - future iterations (currently only 1 supported)
   * @returns {Array<number>} array indexes to drop
   */
  static getDropableColumns(features, n_in, n_out) {
    if (n_out !== 1) throw new RangeError('Only 1 future iteration supported');
    const cols = features + 1;
    const total_cols = cols * n_in + cols * n_out;
    // console.log({ cols, total_cols });
    return range(total_cols - cols +1, total_cols);
  }
  /**
   * Reshape input to be [samples, time steps, features]
   * @example
   * @override 
   * LSTMTimeSeries.getTimeseriesShape([ 
      [ [ 1 ], [ 2 ], [ 3 ] ],
      [ [ 2 ], [ 3 ], [ 4 ] ],
      [ [ 3 ], [ 4 ], [ 5 ] ],
      [ [ 4 ], [ 5 ], [ 6 ] ],
      [ [ 5 ], [ 6 ], [ 7 ] ],
      [ [ 6 ], [ 7 ], [ 8 ] ], 
    ]) //=> [6, 1, 3,]
   * @param {Array<Array<number>} x_timeseries - dataset array of values
   * @return {Array<Array<number>>} returns proper timeseries forecasting shape
   */
  static getTimeseriesShape(x_timeseries) {
    const time_steps = (this.settings && this.settings.lookback) ? this.settings.lookback : 1;
    const xShape = this.getInputShape(x_timeseries);
    const _samples = xShape[ 0 ];
    const _timeSteps = time_steps;
    const _features = (this.settings && this.settings.features) ? this.settings.features : xShape[ 1 ];
    // reshape input to be 3D [samples, timesteps, features]
    // train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    const newShape = [ _samples, _timeSteps, _features, ];
    // console.log({newShape})
    return newShape;
  }
  /**
   * Returns data for predicting values
   * @param timeseries 
   * @param look_back 
   * @override 
   */
  static getTimeseriesDataSet(timeseries, look_back) {
    const lookBack = look_back || this.settings.lookback;
    const matrices = LSTMMultivariateTimeSeries.createDataset.call(this, timeseries, lookBack);
    const x_matrix = matrices[ 0 ];
    const y_matrix_m = LSTMMultivariateTimeSeries.reshape(matrices[ 1 ], [ matrices[ 1 ].length, 1 ]);
    const timeseriesShape = LSTMMultivariateTimeSeries.getTimeseriesShape.call(this,x_matrix);
    // const timeseriesShape = LSTMMultivariateTimeSeries.getTimeseriesShape(x_matrix);
    const x_matrix_timeseries = LSTMMultivariateTimeSeries.reshape(x_matrix, timeseriesShape);
    // const x_matrix_timeseries = LSTMMultivariateTimeSeries.reshape(x_matrix, [x_matrix.length, lookBack, ]);
    const xShape = LSTMMultivariateTimeSeries.getInputShape(x_matrix_timeseries);
    const y_matrix = pivotVector(y_matrix_m)[ y_matrix_m[0].length - 1 ];

    const yShape = LSTMMultivariateTimeSeries.getInputShape(y_matrix_m);
    return {
      yShape,
      xShape,
      y_matrix,
      x_matrix: x_matrix_timeseries,
    };
  }
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      stateful: true,
      mulitpleTimeSteps:false,
      lookback: 1,
      features: undefined,
      outputs:1,
      learningRate: 0.1,
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 1,
        shuffle: false,
      },
    }, options);
    super(config, properties);
    this.createDataset = LSTMMultivariateTimeSeries.createDataset;
    this.seriesToSupervised = LSTMMultivariateTimeSeries.seriesToSupervised;
    this.getDropableColumns = LSTMMultivariateTimeSeries.getDropableColumns;
    this.drop = LSTMMultivariateTimeSeries.drop;
    this.getTimeseriesShape = LSTMMultivariateTimeSeries.getTimeseriesShape;
    this.getTimeseriesDataSet = LSTMMultivariateTimeSeries.getTimeseriesDataSet;
    return this;
  }
  /**
   * Adds dense layers to tensorflow classification model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   * @param {Array<Array<number>>} x_test - validation data independent variables
   * @param {Array<Array<number>>} y_test - validation data dependent variables
   */
  generateLayers(x_matrix, y_matrix, layers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    this.yShape = yShape;
    this.xShape = xShape;
    const lstmLayers = [];
    const denseLayers = [];
    if (layers) {
      if(layers.lstmLayers && layers.lstmLayers.length) lstmLayers.push(...layers.lstmLayers);
      if(layers.denseLayers && layers.denseLayers.length) denseLayers.push(...layers.denseLayers);
   
    } else {
      const inputShape = [ xShape[ 1 ], xShape[ 2 ], ];
      // console.log('default timeseries', { inputShape, xShape, yShape, });
      // model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
      // model.add(Dense(1))

      lstmLayers.push({ units: 10, inputShape,  });
      denseLayers.push({ units: yShape[1], });
    }
    // console.log('lstmLayers',lstmLayers)
    // console.log('denseLayers',denseLayers)
    if (lstmLayers.length) {
      lstmLayers.forEach(layer => {
        this.model.add(this.tf.layers.lstm(layer));
      });
    }
    if (denseLayers.length) {
      denseLayers.forEach(layer => {
        this.model.add(this.tf.layers.dense(layer));
      });
    }
    this.layers = {
      lstmLayers,
      denseLayers,
    };
    // this.settings.compile.optimizer = sgdoptimizer;
  }
  /**
   * @override 
   * @param x_timeseries 
   * @param y_timeseries 
   * @param layers 
   * @param x_test 
   * @param y_test 
   */
  async train(x_timeseries, y_timeseries, layers, x_test, y_test) {
    let xShape;
    let yShape;
    let x_matrix;
    let y_matrix;
    const look_back = this.settings.lookback;
    if (y_timeseries) {
      x_matrix = x_timeseries;
      y_matrix = y_timeseries;
      xShape = this.getInputShape(x_matrix);
      yShape = this.getInputShape(y_matrix);
    } else {
      const matrices = this.createDataset(x_timeseries, look_back);
      // console.log({matrices})
      x_matrix = matrices[ 0 ];
      y_matrix = this.reshape(matrices[ 1 ], [ matrices[ 1 ].length, 1 ]);
      xShape = this.getInputShape(x_matrix);
      yShape = this.getInputShape(y_matrix);
    }
    //_samples, _timeSteps, _features
    const timeseriesShape = this.getTimeseriesShape(x_matrix);
    // console.log({
    //   timeseriesShape, yShape, xShape,
    //   // y_matrix, x_matrix,
    // });
    const x_matrix_timeseries = LSTMMultivariateTimeSeries.reshape(x_matrix, timeseriesShape);
    // console.log('x_matrix_timeseries',JSON.stringify(x_matrix_timeseries));
    // console.log('x_matrix',JSON.stringify(x_matrix));
    // console.log('y_matrix',JSON.stringify(y_matrix));
    const xs = this.tf.tensor(x_matrix_timeseries, timeseriesShape);
    const ys = this.tf.tensor(y_matrix, yShape);
    this.xShape = timeseriesShape;
    this.yShape = yShape;
    this.model = this.tf.sequential();
    this.generateLayers.call(this, x_matrix_timeseries, y_matrix, layers || this.layers, x_test, y_test);
    this.model.compile(this.settings.compile);
    if (x_test && y_test) {
      this.settings.fit.validation_data = [ x_test, y_test, ];
    }
    await this.model.fit(xs, ys, this.settings.fit);
    // this.model.summary();
    xs.dispose();
    ys.dispose();
    return this.model;
  }
}

/**
 * Returns an array of vectors as an array of arrays (from modelscript)
 * @example
const vectors = [ [1,2,3], [1,2,3], [3,3,4], [3,3,3] ];
const arrays = pivotVector(vectors); // => [ [1,2,3,3], [2,2,3,3], [3,3,4,3] ];
 * @memberOf util
 * @param {Array[]} vectors 
 * @returns {Array[]}
 * @ignore
 * @see {https://github.com/repetere/modelscript/blob/master/src/util.js}
 */
/* istanbul ignore next */
export function pivotVector(vectors=[]) {
  /* istanbul ignore next */
  return vectors.reduce((result, val, index/*, arr*/) => {
    val.forEach((vecVal, i) => {
      (index === 0)
        ? (result.push([vecVal,]))
        : (result[ i ].push(vecVal));
    });
    return result;
  }, []);
} 
