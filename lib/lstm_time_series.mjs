import { BaseNeuralNetwork, } from './deep_learning';
import range from 'lodash.range';

/**
 * Long Short Term Memory Time Series with Tensorflow
 * @class LSTMTimeSeries
 * @implements {BaseNeuralNetwork}
 */
export class LSTMTimeSeries extends BaseNeuralNetwork {
  /**
   * Creates dataset data
   * @example
   * LSTMTimeSeries.createDataset([ [ 1, ], [ 2, ], [ 3, ], [ 4, ], [ 5, ], [ 6, ], [ 7, ], [ 8, ], [ 9, ], [ 10, ], ], 3) // => 
      //  [ 
      //    [ 
      //      [ [ 1 ], [ 2 ], [ 3 ] ],
      //      [ [ 2 ], [ 3 ], [ 4 ] ],
      //      [ [ 3 ], [ 4 ], [ 5 ] ],
      //      [ [ 4 ], [ 5 ], [ 6 ] ],
      //      [ [ 5 ], [ 6 ], [ 7 ] ],
      //      [ [ 6 ], [ 7 ], [ 8 ] ], 
      //   ], //x_matrix
      //   [ [ 4 ], [ 5 ], [ 6 ], [ 7 ], [ 8 ], [ 9 ] ] //y_matrix
      // ]
   * @param {Array<Array<number>} dataset - array of values
   * @param {Number} look_back - number of values in each feature 
   * @return {[Array<Array<number>>,Array<number>]} returns x matrix and y matrix for model trainning
   */
  /* istanbul ignore next */
  static createDataset(dataset=[], look_back = 1) { 
    const dataX = [];
    const dataY = [];
    for (let index in range(dataset.length - look_back - 1)) {
      let i = parseInt(index);
      let a = dataset.slice(i, i + look_back);
      dataX.push(a);
      dataY.push(dataset[ i + look_back ]);
    }
    return [dataX, dataY, ];
  }
  /**
   * Reshape input to be [samples, time steps, features]
   * @example
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
    const time_steps = this.settings.timeSteps;
    const xShape = this.getInputShape(x_timeseries);
    const _samples = xShape[ 0 ];
    const _timeSteps = time_steps;
    const _features = xShape[ 1 ];
    const newShape = (this.settings.mulitpleTimeSteps || this.settings.stateful)
      ? [_samples,  _features, _timeSteps, ]
      : [ _samples, _timeSteps, _features, ];
    // console.log({newShape})
    return newShape;
  }
  /**
   * Returns data for predicting values
   * @param timeseries 
   * @param look_back 
   */
  static getTimeseriesDataSet(timeseries, look_back) {
    const lookBack = look_back || this.settings.lookBack;
    const matrices = LSTMTimeSeries.createDataset.call(this, timeseries, lookBack);
    const x_matrix = matrices[ 0 ];
    const y_matrix = matrices[ 1 ];
    // const timeseriesShape = LSTMTimeSeries.getTimeseriesShape.call(this,x_matrix);
    const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, [x_matrix.length, lookBack, ]);
    const xShape = BaseNeuralNetwork.getInputShape(x_matrix_timeseries);
    const yShape = BaseNeuralNetwork.getInputShape(y_matrix);
    return {
      yShape,
      xShape,
      y_matrix,
      x_matrix:x_matrix_timeseries,
    };
  }
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      type: 'simple',
      stateful:false,
      stacked: false,
      mulitpleTimeSteps:false,
      lookBack:1,
      batchSize:1,
      timeSteps:1,
      learningRate:0.1,
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 1,
      },
    }, options);
    super(config, properties);
    this.createDataset = LSTMTimeSeries.createDataset;
    this.getTimeseriesDataSet = LSTMTimeSeries.getTimeseriesDataSet;
    this.getTimeseriesShape = LSTMTimeSeries.getTimeseriesShape;
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
    // const sgdoptimizer = this.tf.train.sgd(this.settings.learningRate);
    const lstmLayers = [];
    const rnnLayers = [];
    const denseLayers = [];
    /* istanbul ignore next */
    if (layers) {
      if(layers.lstmLayers && layers.lstmLayers.length) lstmLayers.push(...layers.lstmLayers);
      if(layers.rnnLayers && layers.rnnLayers.length) rnnLayers.push(...layers.rnnLayers);
      if(layers.denseLayers && layers.denseLayers.length) denseLayers.push(...layers.denseLayers);
    } else if (this.settings.stateful) {
      const batchInputShape = [this.settings.fit.batchSize, this.settings.lookBack, 1, ];
      rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  returnSequences:true, });
      rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  });
      denseLayers.push({ units: yShape[1], });
    // } else if(this.settings.stacked) {
    //   lstmLayers.push({ units: 4, inputShape: [ 1, this.settings.lookBack ], });
    //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    //   denseLayers.push({ units: yShape[1], });
    } else {
      const inputShape= [1, this.settings.lookBack, ];
      // console.log('default timeseries', { inputShape, xShape, yShape ,  });
      lstmLayers.push({ units:4, inputShape,  });
      denseLayers.push({ units: yShape[1], });
    }
    // console.log('lstmLayers',lstmLayers)
    // console.log('denseLayers',denseLayers)
    if (lstmLayers.length) {
      lstmLayers.forEach(layer => {
        this.model.add(this.tf.layers.lstm(layer));
      });
    }
    if (rnnLayers.length) {
      /* istanbul ignore next */
      rnnLayers.forEach(layer => {
        this.model.add(this.tf.layers.simpleRNN(layer));
      });
    }
    if (denseLayers.length) {
      denseLayers.forEach(layer => {
        this.model.add(this.tf.layers.dense(layer));
      });
    }
    this.layers = {
      lstmLayers,
      rnnLayers,
      denseLayers,
    };
    // this.settings.compile.optimizer = sgdoptimizer;
  }
  async train(x_timeseries, y_timeseries, layers, x_test, y_test) {
    let yShape;
    let x_matrix;
    let y_matrix;
    const look_back = this.settings.lookBack;
    if (y_timeseries) {
      x_matrix = x_timeseries;
      y_matrix = y_timeseries;
    } else {
      const matrices = this.createDataset(x_timeseries, look_back);
      x_matrix = matrices[ 0 ];
      y_matrix = matrices[ 1 ];
      yShape = this.getInputShape(y_matrix);
    }
    //_samples, _timeSteps, _features
    const timeseriesShape = this.getTimeseriesShape(x_matrix);
    const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
    const xs = this.tf.tensor(x_matrix_timeseries, timeseriesShape);
    const ys = this.tf.tensor(y_matrix, yShape);
    this.xShape = timeseriesShape;
    this.yShape = yShape;
    this.model = this.tf.sequential();
    this.generateLayers.call(this, x_matrix_timeseries, y_matrix, layers || this.layers, x_test, y_test);
    this.model.compile(this.settings.compile);
    if (this.settings.stateful) {
      this.settings.fit.shuffle = false;
    }
    await this.model.fit(xs, ys, this.settings.fit);
    // this.model.summary();
    xs.dispose();
    ys.dispose();
    return this.model;
  }
  calculate(x_matrix) {
    const timeseriesShape = this.getTimeseriesShape(x_matrix);
    const input_matrix = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
    return super.calculate(input_matrix);
  }
  async predict(input_matrix, options) {
    if (this.settings.stateful && input_matrix.length > 1) {
      return Promise.all(input_matrix.map(input=>super.predict([input, ], options))) ;
    } else {
      return super.predict(input_matrix, options);
    }
  }
}