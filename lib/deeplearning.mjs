import { TensorScriptModelInterface, } from '@tensorscript/core';

/**
 * Deep Learning with Tensorflow
 * @class BaseNeuralNetwork
 * @implements {TensorScriptModelInterface}
 */
export class BaseNeuralNetwork extends TensorScriptModelInterface {
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
   * @return {Object} returns trained tensorflow model 
   */
  async train(x_matrix, y_matrix, layers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    const xs = this.tf.tensor(x_matrix, xShape);
    const ys = this.tf.tensor(y_matrix, yShape);
    this.xShape = xShape;
    this.yShape = yShape;
    this.model = this.tf.sequential();
    this.generateLayers.call(this, x_matrix, y_matrix, layers || this.layers);
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
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(input_matrix) {
    if (!input_matrix || Array.isArray(input_matrix)===false) throw new Error('invalid input matrix');
    const predictionInput = (Array.isArray(input_matrix[ 0 ]))
      ? input_matrix
      : [
        input_matrix,
      ];
    const predictionTensor = this.tf.tensor(predictionInput);
    const prediction = this.model.predict(predictionTensor);
    predictionTensor.dispose();
    return prediction;
  }
}