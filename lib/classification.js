import { BaseNeuralNetwork, } from './deep_learning';

/**
 * Deep Learning Classification with Tensorflow
 * @class DeepLearningClassification
 * @implements {BaseNeuralNetwork}
 */
export class DeepLearningClassification extends BaseNeuralNetwork{
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