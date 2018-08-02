import * as tensorflow from '@tensorflow/tfjs';
/* fix for rollup */
/* istanbul ignore next */
const tf = (tensorflow.default) ? tensorflow.default : tensorflow;
/**
 * Base class for tensorscript models
 * @interface TensorScriptModelInterface
 * @property {Object} settings - tensorflow model hyperparameters
 * @property {Object} model - tensorflow model
 * @property {Object} tf - tensorflow / tensorflow-node / tensorflow-node-gpu
 * @property {Function} reshape - static reshape array function
 * @property {Function} getInputShape - static TensorScriptModelInterface
 */
export class TensorScriptModelInterface {
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
export function size (x) {
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
export function _reshape(array, sizes) {
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
export class DimensionError extends RangeError {
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
export function flatten (array) {
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