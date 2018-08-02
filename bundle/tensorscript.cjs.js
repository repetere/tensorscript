'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var tensorflow = require('@tensorflow/tfjs');
var tensorflow__default = _interopDefault(tensorflow);
var range = _interopDefault(require('lodash.range'));

/* fix for rollup */
/* istanbul ignore next */
const tf = (tensorflow__default) ? tensorflow__default : tensorflow;
/**
 * Base class for tensorscript models
 * @interface TensorScriptModelInterface
 * @property {Object} settings - tensorflow model hyperparameters
 * @property {Object} model - tensorflow model
 * @property {Object} tf - tensorflow / tensorflow-node / tensorflow-node-gpu
 * @property {Function} reshape - static reshape array function
 * @property {Function} getInputShape - static TensorScriptModelInterface
 */
class TensorScriptModelInterface {
  /**
   * @param {Object} options - tensorflow model hyperparameters
   * @param {Object} customTF - custom, overridale tensorflow / tensorflow-node / tensorflow-node-gpu
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties = {}) {
    /** @type {Object} */
    this.settings = options;
    /** @type {Object} */
    this.model = properties.model;
    /** @type {Object} */
    this.tf = properties.tf || tf;
    /** @type {Function} */
    this.reshape = TensorScriptModelInterface.reshape;
    /** @type {Function} */
    this.getInputShape = TensorScriptModelInterface.getInputShape;
    return this;
  }
  /**
   * Reshapes an array
   * @function
   * @example 
   * const array = [ 0, 1, 1, 0, ];
   * const shape = [2,2];
   * TensorScriptModelInterface.reshape(array,shape) // => 
   * [
   *   [ 0, 1, ],
   *   [ 1, 0, ],
   * ];
   * @param {Array<number>} array - input array 
   * @param {Array<number>} shape - shape array 
   * @return {Array<Array<number>>} returns a matrix with the defined shape
   */
  /* istanbul ignore next */
  static reshape(array, shape) {
    const flatArray = flatten(array);
   

    function product (arr) {
      return arr.reduce((prev, curr) => prev * curr);
    }
  
    if (!Array.isArray(array) || !Array.isArray(shape)) {
      throw new TypeError('Array expected');
    }
  
    if (shape.length === 0) {
      throw new DimensionError(0, product(size(array)), '!=');
    }
    let newArray;
    let totalSize = 1;
    const rows = shape[ 0 ];
    for (let sizeIndex = 0; sizeIndex < shape.length; sizeIndex++) {
      totalSize *= shape[sizeIndex];
    }
  
    if (flatArray.length !== totalSize) {
      throw new DimensionError(
        product(shape),
        product(size(array)),
        '!='
      );
    }
  
    try {
      newArray = _reshape(flatArray, shape);
    } catch (e) {
      if (e instanceof DimensionError) {
        throw new DimensionError(
          product(shape),
          product(size(array)),
          '!='
        );
      }
      throw e;
    }
    if (newArray.length !== rows) throw new SyntaxError(`specified shape (${shape}) is compatible with input array or length (${array.length})`);

    // console.log({ newArray ,});
    return newArray;
  }
  /**
   * Returns the shape of an input matrix
   * @function
   * @example 
   * const input = [
   *   [ 0, 1, ],
   *   [ 1, 0, ],
   * ];
   * TensorScriptModelInterface.getInputShape(input) // => [2,2]
   * @see {https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array}
   * @param {Array<Array<number>>} matrix - input matrix 
   * @return {Array<number>} returns the shape of a matrix (e.g. [2,2])
   */
  static getInputShape(matrix=[]) {
    if (Array.isArray(matrix) === false || !matrix[ 0 ] || !matrix[ 0 ].length || Array.isArray(matrix[ 0 ]) === false) throw new TypeError('input must be a matrix');
    const dim = [];
    const x_dimensions = matrix[ 0 ].length;
    let vectors = matrix;
    matrix.forEach(vector => {
      if (vector.length !== x_dimensions) throw new SyntaxError('input must have the same length in each row');
    });
    for (;;) {
      dim.push(vectors.length);
      if (Array.isArray(vectors[0])) {
        vectors = vectors[0];
      } else {
        break;
      }
    }
    return dim;
  }
  /**
   * Asynchronously trains tensorflow model, must be implemented by tensorscript class
   * @abstract 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @return {Object} returns trained tensorflow model 
   */
  train(x_matrix, y_matrix) {
    throw new ReferenceError('train method is not implemented');
  }
  /**
   * Predicts new dependent variables
   * @abstract 
   * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(matrix) {
    throw new ReferenceError('calculate method is not implemented');
  }
  /**
   * Loads a saved tensoflow / keras model
   * @param {Object} options - tensorflow load model options
   * @return {Object} tensorflow model
   */
  async loadModel(options) {
    this.model = await this.tf.loadModel(options);
    return this.model;
  }
  /**
   * Returns prediction values from tensorflow model
   * @param {Array<Array<number>>|Array<number>} input_matrix - new test independent variables 
   * @param {Boolean} [options.json=true] - return object instead of typed array
   * @param {Boolean} [options.probability=true] - return real values instead of integers
   * @return {Array<number>|Array<Array<number>>} predicted model values
   */
  async predict(input_matrix, options = {}) {
    if (!input_matrix || Array.isArray(input_matrix)===false) throw new Error('invalid input matrix');
    const x_matrix = (Array.isArray(input_matrix[ 0 ]))
      ? input_matrix
      : [
        input_matrix,
      ];
    const config = Object.assign({
      json: true,
      probability: true,
    }, options);
    return this.calculate(x_matrix)
      .data()
      .then(predictions => {
        if (config.json === false) {
          return predictions;
        } else {
          const shape = [x_matrix.length, this.yShape[ 1 ], ];
          const predictionValues = (options.probability === false) ? Array.from(predictions).map(Math.round) : Array.from(predictions);
          return this.reshape(predictionValues, shape);
        }
      })
      .catch(e => {
        throw e; 
      });
  }
}

/**
 * Calculate the size of a multi dimensional array.
 * This function checks the size of the first entry, it does not validate
 * whether all dimensions match. (use function `validate` for that) (from math.js)
 * @param {Array} x
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @ignore
 * @return {Number[]} size
 */
/* istanbul ignore next */
function size (x) {
  let s = [];

  while (Array.isArray(x)) {
    s.push(x.length);
    x = x[0];
  }

  return s;
}
/**
 * Iteratively re-shape a multi dimensional array to fit the specified dimensions (from math.js)
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
function _reshape(array, sizes) {
  // testing if there are enough elements for the requested shape
  var tmpArray = array;
  var tmpArray2;
  // for each dimensions starting by the last one and ignoring the first one
  for (var sizeIndex = sizes.length - 1; sizeIndex > 0; sizeIndex--) {
    var size = sizes[sizeIndex];
    tmpArray2 = [];

    // aggregate the elements of the current tmpArray in elements of the requested size
    var length = tmpArray.length / size;
    for (var i = 0; i < length; i++) {
      tmpArray2.push(tmpArray.slice(i * size, (i + 1) * size));
    }
    // set it as the new tmpArray for the next loop turn or for return
    tmpArray = tmpArray2;
  }
  return tmpArray;
}

/**
 * Create a range error with the message:
 *     'Dimension mismatch (<actual size> != <expected size>)' (from math.js)
 * @param {number | number[]} actual        The actual size
 * @param {number | number[]} expected      The expected size
 * @param {string} [relation='!=']          Optional relation between actual
 *                                          and expected size: '!=', '<', etc.
 * @extends RangeError
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
class DimensionError extends RangeError {
  constructor(actual, expected, relation) {
    /* istanbul ignore next */
    const message = 'Dimension mismatch (' + (Array.isArray(actual) ? ('[' + actual.join(', ') + ']') : actual) + ' ' + ('!=') + ' ' + (Array.isArray(expected) ? ('[' + expected.join(', ') + ']') : expected) +  ')';
    super(message);
  
    this.actual = actual;
    this.expected = expected;
    this.relation = relation;
    // this.stack = (new Error()).stack
    this.message = message;
    this.name = 'DimensionError';
    this.isDimensionError = true;
  }
}

/**
 * Flatten a multi dimensional array, put all elements in a one dimensional
 * array
 * @param {Array} array   A multi dimensional array
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @return {Array}        The flattened array (1 dimensional)
 */
/* istanbul ignore next */
function flatten (array) {
  /* istanbul ignore next */
  if (!Array.isArray(array)) {
    // if not an array, return as is
    /* istanbul ignore next */
    return array;
  }
  let flat = [];
  
  /* istanbul ignore next */
  array.forEach(function callback (value) {
    if (Array.isArray(value)) {
      value.forEach(callback); // traverse through sub-arrays recursively
    } else {
      flat.push(value);
    }
  });

  return flat;
}

/**
 * Deep Learning with Tensorflow
 * @class BaseNeuralNetwork
 * @implements {TensorScriptModelInterface}
 */
class BaseNeuralNetwork extends TensorScriptModelInterface {
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    }, options);
    super(config, properties);
    return this;
  }
  /**
   * Adds dense layers to tensorflow model
   * @abstract 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   */
  generateLayers(x_matrix, y_matrix, layers) {
    throw new ReferenceError('generateLayers method is not implemented');
  }
  /**
   * Asynchronously trains tensorflow model
   * @override
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - array of model dense layer parameters
   * @param {Array<Array<number>>} x_text - validation data independent variables
   * @param {Array<Array<number>>} y_text - validation data dependent variables
   * @return {Object} returns trained tensorflow model 
   */
  async train(x_matrix, y_matrix, layers, x_test, y_test) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    const xs = this.tf.tensor(x_matrix, xShape);
    const ys = this.tf.tensor(y_matrix, yShape);
    this.xShape = xShape;
    this.yShape = yShape;
    this.model = this.tf.sequential();
    this.generateLayers.call(this, x_matrix, y_matrix, layers || this.layers, x_test, y_test);
    this.model.compile(this.settings.compile);
    await this.model.fit(xs, ys, this.settings.fit);
    xs.dispose();
    ys.dispose();
    return this.model;
  }
  /**
   * Predicts new dependent variables
   * @override
   * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
   * @param {Object} options - model prediction options
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(input_matrix, options) {
    if (!input_matrix || Array.isArray(input_matrix)===false) throw new Error('invalid input matrix');
    const predictionInput = (Array.isArray(input_matrix[ 0 ]))
      ? input_matrix
      : [
        input_matrix,
      ];
    const predictionTensor = this.tf.tensor(predictionInput);
    const prediction = this.model.predict(predictionTensor, options);
    predictionTensor.dispose();
    return prediction;
  }
}

/**
 * Deep Learning Regression with Tensorflow
 * @class DeepLearningRegression
 * @implements {BaseNeuralNetwork}
 */
class DeepLearningRegression extends BaseNeuralNetwork {
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object,layerPreference:String}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      layerPreference:'deep',
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    }, options);
    super(config, properties);
    return this;
  }
  /**
   * Adds dense layers to tensorflow regression model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   */
  generateLayers(x_matrix, y_matrix, layers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    const denseLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else if(this.settings.layerPreference==='deep') {
      denseLayers.push({ units: xShape[ 1 ], inputShape: [xShape[1],], kernelInitializer: 'randomNormal', activation: 'relu', });
      denseLayers.push({ units: parseInt(Math.ceil(xShape[ 1 ] / 2), 10), kernelInitializer: 'randomNormal', activation: 'relu', });
      denseLayers.push({ units: yShape[ 1 ], kernelInitializer: 'randomNormal', });
    } else {
      denseLayers.push({ units: (xShape[ 1 ] * 2), inputShape: [xShape[1],], kernelInitializer: 'randomNormal', activation: 'relu', });
      denseLayers.push({ units: yShape[ 1 ], kernelInitializer: 'randomNormal', });
    }
    this.layers = denseLayers;
    denseLayers.forEach(layer => {
      this.model.add(this.tf.layers.dense(layer));
    });
  }
}

/**
 * Deep Learning Classification with Tensorflow
 * @class DeepLearningClassification
 * @implements {BaseNeuralNetwork}
 */
class DeepLearningClassification extends BaseNeuralNetwork{
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      compile: {
        loss: 'categoricalCrossentropy',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    }, options);
    super(config, properties);
    return this;
  }
  /**
   * Adds dense layers to tensorflow classification model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   */
  generateLayers(x_matrix, y_matrix, layers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    this.yShape = yShape;
    this.xShape = xShape;
    const denseLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else {
      denseLayers.push({ units: (xShape[ 1 ] * 2), inputDim: xShape[1],  activation: 'relu', });
      denseLayers.push({ units: yShape[ 1 ], activation: 'softmax', });
    }
    this.layers = denseLayers;
    denseLayers.forEach(layer => {
      this.model.add(this.tf.layers.dense(layer));
    });
  }
}

/**
 * Logistic Regression Classification with Tensorflow
 * @class LogisticRegression
 * @implements {BaseNeuralNetwork}
 */
class LogisticRegression extends BaseNeuralNetwork {
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
      layers: [],
      type:'simple',
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'rmsprop',
      },
      fit: {
        epochs: 100,
        batchSize: 5,
      },
    }, options);
    super(config, properties);
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
  generateLayers(x_matrix, y_matrix, layers, x_test, y_test) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    this.yShape = yShape;
    this.xShape = xShape;
    const denseLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else if (this.settings.type==='class') { 
      denseLayers.push({ units: 1, inputDim:  xShape[ 1 ], activation: 'sigmoid', });
      this.settings.compile.loss = 'binaryCrossentropy';
    } else if (this.settings.type === 'l1l2') { 
      const kernelRegularizer = this.tf.regularizers.l1l2({ l1: 0.01, l2: 0.01, });
      denseLayers.push({ units: 1, inputDim:  xShape[ 1 ], activation: 'sigmoid', kernelRegularizer, });
      this.settings.compile.loss = 'binaryCrossentropy';
    } else {
      denseLayers.push({ units: 1, inputShape: [xShape[1], ], });
    }
    this.layers = denseLayers;
    denseLayers.forEach(layer => {
      this.model.add(this.tf.layers.dense(layer));
    });
    /* istanbul ignore next */
    if (x_test && y_test) {
      this.settings.fit.validationData = [x_test, y_test];
    }
  }
}

/**
 * Mulitple Linear Regression with Tensorflow
 * @class MultipleLinearRegression
 * @implements {BaseNeuralNetwork}
 */
class MultipleLinearRegression extends BaseNeuralNetwork {
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  /* istanbul ignore next */
  constructor(options = {}, properties = {}) {
    const config = Object.assign({
      layers: [],
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'sgd',
      },
      fit: {
        epochs: 500,
        batchSize: 5,
      },
    }, options);
    super(config, properties);
    return this;
  }
  /**
   * Adds dense layers to tensorflow regression model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   */
  generateLayers(x_matrix, y_matrix, layers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    const denseLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else {
      denseLayers.push({ units: yShape[1], inputShape: [xShape[1],], });
    }
    this.layers = denseLayers;
    denseLayers.forEach(layer => {
      this.model.add(this.tf.layers.dense(layer));
    });
  }
}

/**
 * Long Short Term Memory Time Series with Tensorflow
 * @class LSTMTimeSeries
 * @implements {BaseNeuralNetwork}
 */
class LSTMTimeSeries extends BaseNeuralNetwork {
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

/**
 * Long Short Term Memory Multivariate Time Series with Tensorflow
 * @class LSTMMultivariateTimeSeries
 * @extends {LSTMTimeSeries}
 */
class LSTMMultivariateTimeSeries extends LSTMTimeSeries {
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
    const original_dropped_c_columns = [ 0, droppedColumns[ 0 ].length - 1, ];
    
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
function pivotVector(vectors=[]) {
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

exports.TensorScriptModelInterface = TensorScriptModelInterface;
exports.BaseNeuralNetwork = BaseNeuralNetwork;
exports.DeepLearningRegression = DeepLearningRegression;
exports.DeepLearningClassification = DeepLearningClassification;
exports.LogisticRegression = LogisticRegression;
exports.MultipleLinearRegression = MultipleLinearRegression;
exports.LSTMTimeSeries = LSTMTimeSeries;
exports.LSTMMultivariateTimeSeries = LSTMMultivariateTimeSeries;
