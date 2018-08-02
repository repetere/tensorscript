import chai from 'chai';
import sinon from 'sinon';
import * as ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { BaseNeuralNetwork, } from '../../index.mjs';
const expect = chai.expect;
const independentVariables = [
  'CRIM',
  'ZN',
  'INDUS',
  'CHAS',
  'NOX',
  'RM',
  'AGE',
  'DIS',
  'RAD',
  'TAX',
  'PTRATIO',
  'B',
  'LSTAT',
];
const dependentVariables = [
  'MEDV',
];
const columns = independentVariables.concat(dependentVariables);
let housingDataCSV;
let DataSet;

chai.use(sinonChai);
chai.use(chaiAsPromised);
function scaleColumnMap(columnName) {
  return {
    name: columnName,
    options: {
      strategy: 'scale',
      scaleOptions: {
        strategy:'standard',
      },
    },
  };
}
/** @test {BaseNeuralNetwork} */
describe('BaseNeuralNetwork', function () {
  this.timeout(10000);
  before(async function () {
    housingDataCSV = await ms.csv.loadCSV('./test/mock/data/boston_housing_data.csv');
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: columns.map(scaleColumnMap),
      returnData:false,
    });
  });
  /** @test {BaseNeuralNetwork#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new BaseNeuralNetwork();
      const MLRConfigured = new BaseNeuralNetwork({ test: 'prop', });
      expect(BaseNeuralNetwork).to.be.a('function');
      expect(MLR).to.be.instanceOf(BaseNeuralNetwork);
      expect(MLRConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {BaseNeuralNetwork#generateLayers} */
  describe('generateLayers', () => {
    it('should throw an error if generateLayers method is not implemented', () => {
      class NN extends BaseNeuralNetwork{
        generateLayers(x, y, layers) {
          return true;
        }
      }
      const TS = new BaseNeuralNetwork();
      const TSNN = new NN();
      expect(TS.generateLayers).to.be.a('function');
      expect(TS.generateLayers.bind(null)).to.throw('generateLayers method is not implemented');
      expect(TSNN.generateLayers).to.be.a('function');
      expect(TSNN.generateLayers.bind(null)).to.be.ok;
    });
  });
  /** @test {BaseNeuralNetwork#train} */
  describe('train', () => {
    it('should train a NN', async function () {
      const NN = new BaseNeuralNetwork();
      const x = [];
      const y = [];
      const layers = [];
      const tf = {
        tensor: () => ({ 
          dispose: () => { },
        }),
        sequential: () => ({
          compile: () => true,
          fit: () => true,
        }),
      };
      const settings = {};
      function getInputShape() { }
      function generateLayers() { }
      const trainedModel = await NN.train.call({
        getInputShape,
        generateLayers,
        tf,
        settings,
      }, x, y, layers);
      const trainedModel2 = await NN.train.call({
        getInputShape,
        generateLayers,
        tf,
        settings,
        layers:[],
      }, x, y);
      expect(trainedModel).to.be.an('object');
      expect(trainedModel2).to.be.an('object');
    });
  });
  /** @test {BaseNeuralNetwork#calculate} */
  describe('calculate', () => {
    it('should throw an error if input is invalid', () => {
      const NN = new BaseNeuralNetwork();
      expect(NN.calculate).to.be.a('function');
      expect(NN.calculate.bind()).to.throw(/invalid input matrix/);
      expect(NN.calculate.bind(null, 'invalid')).to.throw(/invalid input matrix/);
    });
    it('should train a NN', async function () {
      const NN = new BaseNeuralNetwork();
      const x = [1, 2, 3, ];
      const x2 = [[1, 2, 3, ], [1, 2, 3, ], ];
      const tf = {
        tensor: () => ({ 
          dispose: () => { },
        }),
        sequential: () => ({
          compile: () => true,
          fit: () => true,
        }),
      };
      const model = {
        predict: () => true,
      };
      const prediction = NN.calculate.call({
        tf,
        model,
      }, x);
      const prediction2 = NN.calculate.call({
        tf,
        model,
      }, x2);
      expect(prediction).to.be.true;
      expect(prediction2).to.be.true;
    });
  });
});