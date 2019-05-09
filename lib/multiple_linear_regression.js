import { BaseNeuralNetwork, } from './deep_learning';

/**
 * Mulitple Linear Regression with Tensorflow
 * @class MultipleLinearRegression
 * @implements {BaseNeuralNetwork}
 */
export class MultipleLinearRegression extends BaseNeuralNetwork {
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