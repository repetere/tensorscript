import chai from 'chai';
// import sinon from 'sinon';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { TensorScriptModelInterface, } from '../../index.mjs';
const expect = chai.expect;
chai.use(sinonChai);
chai.use(chaiAsPromised);

/** @test {TensorScriptModelInterface} */
describe('TensorScriptModelInterface', () => {
  /** @test {TensorScriptModelInterface#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const TSM = new TensorScriptModelInterface();
      const TSMConfigured = new TensorScriptModelInterface({ test: 'prop', });
      expect(TensorScriptModelInterface).to.be.a('function');
      expect(TSM).to.be.instanceOf(TensorScriptModelInterface);
      expect(TSMConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {TensorScriptModelInterface#reshape} */
  describe('reshape', () => {
    it('should export a static method', () => {
      expect(TensorScriptModelInterface.reshape).to.be.a('function');
    });
    it('should reshape an array into a matrix', () => {
      const array = [1, 0, 0, 1, ];
      const shape = [2, 2, ];
      const matrix = [
        [1, 0,],
        [0, 1,],
      ];
      const result = TensorScriptModelInterface.reshape(array, shape);
      expect(result).to.eql(matrix);
      // expect(TensorScriptModelInterface.reshape.bind(null, array, [1, 2, ])).to.throw(/specified shape/);
    });
    it('should reshape multiple dimensions', () => {
      const array = [ 1, 1, 0, 1, 1, 0, ];
      const shape = [ 2, 3, 1, ];
      const matrix = [
        [ [1], [1], [0], ],
        [ [1], [1], [0], ],
      ];
      const result = TensorScriptModelInterface.reshape(array, shape);
      // console.log({ result });
    });
  });
  /** @test {TensorScriptModelInterface#getInputShape} */
  describe('getInputShape', () => {
    it('should export a static method', () => {
      expect(TensorScriptModelInterface.getInputShape).to.be.a('function');
    });
    it('should return the shape of a matrix', () => {
      const matrix = [
        [1, 0,],
        [0, 1,],
      ];
      const matrix2 = [
        [1, 0,],
        [1, 0,],
        [1, 0,],
        [1, 0,],
        [1, 0,],
        [1, 0,],
        [0, 1,],
      ];
      const matrix3 = [
        [1, 0, 4, 5,],
        [1, 0, 4, 5,],
        [1, 0, 4, 5,],
        [1, 0, 4, 5,],
        [1, 0, 4, 5,],
      ];
      const matrix4 = [
        [1, 0, 4, 5,],
        [1, 0, 4,],
        [1, 0, 4, 5,],
      ];
      const matrix5 = [
        [[1, ], [1, ], [0, ], ],
        [[1, ], [1, ], [0, ], ],
        [[1, ], [1, ], [0, ], ],
        [[1, ], [1, ], [0, ], ],
        [[1, ], [1, ], [0, ], ],
        [[1, ], [1, ], [0, ], ],
      ];
      TensorScriptModelInterface.getInputShape(matrix5);
      expect(TensorScriptModelInterface.getInputShape(matrix)).to.eql([2, 2,]);
      expect(TensorScriptModelInterface.getInputShape(matrix2)).to.eql([7, 2,]);
      expect(TensorScriptModelInterface.getInputShape(matrix3)).to.eql([5, 4,]);
      expect(TensorScriptModelInterface.getInputShape.bind(null, matrix4)).to.throw(/input must have the same length in each row/);
      expect(TensorScriptModelInterface.getInputShape(matrix5)).to.eql([6, 3,1]);
    });
    it('should throw an error if input is not a matrix', () => {
      expect(TensorScriptModelInterface.getInputShape.bind()).to.throw(/must be a matrix/);
    });
  });
  /** @test {TensorScriptModelInterface#train} */
  describe('train', () => {
    it('should throw an error if train method is not implemented', () => {
      class MLR extends TensorScriptModelInterface{
        train(x, y) {
          return true;
        }
      }
      const TSM = new TensorScriptModelInterface();
      const TSMMLR = new MLR();
      expect(TSM.train).to.be.a('function');
      expect(TSM.train.bind(null)).to.throw('train method is not implemented');
      expect(TSMMLR.train).to.be.a('function');
      expect(TSMMLR.train.bind(null)).to.be.ok;
    });
  });
  /** @test {TensorScriptModelInterface#calculate} */
  describe('calculate', () => {
    it('should throw an error if calculate method is not implemented', () => {
      class MLR extends TensorScriptModelInterface{
        calculate(x, y) {
          return true;
        }
      }
      const TSM = new TensorScriptModelInterface();
      const TSMMLR = new MLR();
      expect(TSM.calculate).to.be.a('function');
      expect(TSM.calculate.bind(null)).to.throw('calculate method is not implemented');
      expect(TSMMLR.calculate).to.be.a('function');
      expect(TSMMLR.calculate.bind(null)).to.be.ok;
    });
  });
  /** @test {TensorScriptModelInterface#predict} */
  describe('predict', () => {
    class MLR extends TensorScriptModelInterface{
      calculate(x) {
        this.yShape = [100, 2,];
        return {
          data: () => new Promise((resolve) => {
            const predictions = new Float32Array([21.41, 31.74, 41.01, 51.53,]);
            resolve(predictions);
          }),
        };
      }
    }
    it('should throw an error if input is invalid', async function () {
      const TSMMLR = new MLR();
      try {
        const predictPromise = await TSMMLR.predict();
        expect(predictPromise).to.not.exist;
      } catch (e) {
        expect(e).to.be.an('error');
        expect(e).to.match(/invalid input matrix/);
      }
      try {
        const predictPromiseCatch = await TSMMLR.predict([1,]);
        expect(predictPromiseCatch).to.not.exist;
      } catch (e2) {
        expect(e2).to.be.an('error');
        expect(e2).to.match(/Dimension mismatch/);
      }
    });
    it('should return preductions', async function () {
      const TSMMLR = new MLR();
      const input = [
        [1, 2, ],
        [1, 2, ],
      ];
      const predictions = await TSMMLR.predict(input);
      const predictionsRounded = await TSMMLR.predict(input, { probability:false, });
      const predictionsRaw = await TSMMLR.predict(input, { json: false, });
      expect(predictions).to.have.lengthOf(2);
      expect(predictionsRaw).to.be.a('Float32Array');
      predictionsRounded.forEach(predRow => {
        predRow.forEach(pred => {
          expect(Number.isInteger(pred)).to.be.true;
        });
      });
    });
  });
  /** @test {TensorScriptModelInterface#loadModel} */
  describe('loadModel', () => {
    it('should call tensorflow load model and store it', async function () {
      const TSM = new TensorScriptModelInterface({}, {
        tf: {
          loadModel: () => new Promise((resolve, reject) => resolve(true)),
        },
      });
      const loadedModel = await TSM.loadModel();
      expect(loadedModel).to.be.true;
    });
  });
});