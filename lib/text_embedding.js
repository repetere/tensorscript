import { TensorScriptModelInterface, } from './model_interface';
import * as UniversalSentenceEncoder from '@tensorflow-models/universal-sentence-encoder';
let model;
let tokenizer;
/**
 * Text Embedding with Tensorflow Universal Sentence Encoder (USE)
 * @class TextEmbedding
 * @implements {TensorScriptModelInterface}
 */
export class TextEmbedding extends TensorScriptModelInterface {
  /**
   * @param {Object} options - Options for USE
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options = {}, properties) {
    const config = Object.assign({
    }, options);
    super(config, properties);
    return this;
  }
  /**
   * Asynchronously loads Universal Sentence Encoder and tokenizer
   * @override
   * @return {Object} returns loaded UniversalSentenceEncoder model
   */
  async train() {
    const promises = [];
    if (!model) promises.push(UniversalSentenceEncoder.load());
    else promises.push(Promise.resolve(model));
    if (!tokenizer) promises.push(UniversalSentenceEncoder.loadTokenizer());
    else promises.push(Promise.resolve(tokenizer));
    const USE = await Promise.all(promises);
    if (!model) model = USE[ 0 ];
    if (!tokenizer) tokenizer = USE[ 1 ];
    this.model = model;
    this.tokenizer = tokenizer;
    return this.model;
  }
  /**
   * Calculates sentence embeddings
   * @override
   * @param {Array<Array<number>>|Array<number>} input_array - new test independent variables
   * @param {Object} options - model prediction options
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(input_array, options = {}) {
    if (!input_array || Array.isArray(input_array) === false) throw new Error('invalid input array of sentences');
    const embeddings = this.model.embed(input_array);
    return embeddings;
  }
  /**
   * Returns prediction values from tensorflow model
   * @param {Array<string>} input_matrix - array of sentences to embed 
   * @param {Boolean} [options.json=true] - return object instead of typed array
   * @param {Boolean} [options.probability=true] - return real values instead of integers
   * @return {Array<Array<number>>} predicted model values
   */
  async predict(input_array, options = {}) {
    const config = Object.assign({
      json: true,
      probability: true,
    }, options);
    const embeddings = await this.calculate(input_array, options);
    const predictions = await embeddings.data(); 
    if (config.json === false) {
      return predictions;
    } else {
      const shape = [input_array.length, 512, ];
      const predictionValues = (options.probability === false)
        ? Array.from(predictions).map(Math.round)
        : Array.from(predictions);
      return this.reshape(predictionValues, shape);
    }
  }
}