import { BaseNeuralNetwork, } from './deep_learning';

/**
 * Deep Learning Regression with Tensorflow
 * @class DeepLearningRegression
 * @implements {BaseNeuralNetwork}
 */
export class DeepLearningRegression extends BaseNeuralNetwork {
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