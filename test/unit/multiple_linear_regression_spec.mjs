import chai from 'chai';
import sinon from 'sinon';
import ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { MultipleLinearRegression, } from '../../index.mjs';
const expect = chai.expect;
const independentVariables = [ 'sqft', 'bedrooms',];
const dependentVariables = [ 'price', ];
let housingDataCSV;
let input_x;
let DataSet;
let x_matrix;
let y_matrix;
let trainedMLR;
let trainedMLRModel;

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
describe('MultipleLinearRegression', function () {
  this.timeout(10000);
  before(async function () {
    housingDataCSV = await ms.csv.loadCSV('./test/mock/data/portland_housing_data.csv', {
      colParser: {
        sqft: 'number',
        bedrooms: 'number',
        price: 'number',
      },
    });
    /*
    housingdataCSV = [ 
      { sqft: 2104, bedrooms: 3, price: 399900 },
      { sqft: 1600, bedrooms: 3, price: 329900 },
      ...
      { sqft: 1203, bedrooms: 3, price: 239500 } 
    ] 
    */
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: [
        'sqft',
        'bedrooms',
        'price',
      ].map(scaleColumnMap),
      returnData:false,
    });
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    // const y_vector = ms.util.pivotVector(y_matrix)[ 0 ];// not used but just illustrative
    /* x_matrix = [
        [2014, 3],
        [1600, 3],
      ] 
      y_matrix = [
        [399900],
        [329900],
      ] 
      y_vector = [ 399900, 329900]
    */
    trainedMLR = new MultipleLinearRegression();
    trainedMLRModel = await trainedMLR.train(x_matrix, y_matrix);
    input_x = [
      [
        DataSet.scalers.get('sqft').scale(4215),
        DataSet.scalers.get('bedrooms').scale(4),
      ], //549000
      [
        DataSet.scalers.get('sqft').scale(852),
        DataSet.scalers.get('bedrooms').scale(2),
      ], //179900
    ];
  });
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new MultipleLinearRegression();
      const MLRConfigured = new MultipleLinearRegression({ test: 'prop', });
      expect(MultipleLinearRegression).to.be.a('function');
      expect(MLR).to.be.instanceOf(MultipleLinearRegression);
      expect(MLRConfigured.settings.test).to.eql('prop');
    });
  });
  describe('generateLayers', () => {
    it('should generate a classification network', async () => {
      const predictions = await trainedMLR.predict(input_x);
      const shape = trainedMLR.getInputShape(predictions);
      // console.log('nnLR.layers', nnLR.layers);
      // console.log({
      //   predictions,
      //   shape,
      // });
      expect(predictions).to.have.lengthOf(input_x.length);
      expect(trainedMLR.layers).to.have.lengthOf(1);
      const descaledPredictions = predictions.map(DataSet.scalers.get('price').descale);
      expect(descaledPredictions[ 0 ]).to.be.closeTo(630000, 20000);
      expect(descaledPredictions[ 1 ]).to.be.closeTo(190000, 10000);
      return true;
    });
    it('should generate a network from layers', async () => { 
      const nnLRCustom = new MultipleLinearRegression({ type:'custom', });
      await nnLRCustom.train(x_matrix, y_matrix, trainedMLR.layers);
      expect(nnLRCustom.layers).to.have.lengthOf(1);
    });
  });
});