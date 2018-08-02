import chai from 'chai';
// import sinon from 'sinon';
import * as ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { LogisticRegression, } from '../../index.mjs';

const expect = chai.expect;
const independentVariables = [
  'Age',
  'EstimatedSalary',
];
const dependentVariables = [
  'Purchased',
];
let CSVData;
let DataSet;
let x_matrix;
let y_matrix;
let nnLR;
let nnLRClass;
let nnLRReg;
let nnLRModel;
let nnLRClassModel;
let nnLRRegModel;
const encodedAnswers = {
  'yes': [1,],
  'no': [0,],
};
const fit = {
  epochs: 10,
  batchSize: 5,
};
const input_x = [
  [-0.062482849427819266, 0.30083326827486173, ], //0
  [0.7960601198093905, -1.1069168538010206, ], //1
  [0.7960601198093905, 0.12486450301537644, ], //0
  [0.4144854668150751, -0.49102617539282206, ], //0
  [0.3190918035664962, 0.5061301610775946, ], //1
];
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
/** @test {LogisticRegression} */
describe('LogisticRegression', function () {
  this.timeout(120000);
  before(async function () {
    CSVData = await ms.csv.loadCSV('./test/mock/data/social_network_ads.csv');
    DataSet = new ms.DataSet(CSVData);
    const scaledData = DataSet.fitColumns({
      columns: independentVariables.map(scaleColumnMap),
      returnData:true,
    });
    /*
    scaledData = [ 
      { 'User ID': 15624510,
         Gender: 'Male',
         Age: -1.7795687879022388,
         EstimatedSalary: -1.4881825118632386,
         Purchased: 0 },
      { 'User ID': 15810944,
         Gender: 'Male',
         Age: -0.253270175924977,
         EstimatedSalary: -1.458854384319991,
         Purchased: 0 },
      ...
    ];
    */
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    /*
    x_matrix = [
      [ -1.7795687879022388, -1.4881825118632386 ],
      [ -0.253270175924977, -1.458854384319991 ],
      ...
    ]; 
    y_matrix = [
      [ 0 ],
      [ 0 ],
      ...
    ] 
    */
    // console.log({ x_matrix, y_matrix, });

    nnLR = new LogisticRegression({ fit,  });
    nnLRClass = new LogisticRegression({ type: 'class', fit, });
    nnLRReg = new LogisticRegression({ type: 'l1l2', fit, });
    const models = await Promise.all([
      nnLR.train(x_matrix, y_matrix),
      nnLRClass.train(x_matrix, y_matrix),
      nnLRReg.train(x_matrix, y_matrix),
    ]);
    nnLRModel = models[ 0 ];
    nnLRClassModel = models[ 1 ];
    nnLRRegModel = models[ 2 ];
  });
  /** @test {LogisticRegression#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new LogisticRegression();
      const NNConfigured = new LogisticRegression({ test: 'prop', });
      expect(LogisticRegression).to.be.a('function');
      expect(NN).to.be.instanceOf(LogisticRegression);
      expect(NNConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {LogisticRegression#generateLayers} */
  describe('generateLayers', () => {
    it('should generate a classification network', async () => {
      const predictions = await nnLR.predict(input_x);
      const answers = await nnLR.predict(input_x, {
        probability:false,
      });
      const shape = nnLR.getInputShape(predictions);
     
      expect(predictions).to.have.lengthOf(input_x.length);
      expect(nnLR.layers).to.have.lengthOf(1);
      expect(shape).to.eql([5, 1, ]);
      // expect(answers[ 0 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
      return true;
    });
    it('should generate a network from layers', async () => { 
      const nnLRCustom = new LogisticRegression({ type:'custom', fit, });
      await nnLRCustom.train(x_matrix, y_matrix, nnLR.layers);
      expect(nnLRCustom.layers).to.have.lengthOf(1);
    });
    // it('should validate trainning data', async () => { 
    //   const nnLRCustom = new LogisticRegression({ type:'custom', fit, });
    //   await nnLRCustom.train(x_matrix, y_matrix, nnLR.layers, x_matrix, y_matrix);
    //   expect(nnLRCustom.layers).to.have.lengthOf(1);
    // });
  });
});