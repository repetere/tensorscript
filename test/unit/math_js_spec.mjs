import chai from 'chai';
import { TensorScriptModelInterface, size, flatten, } from '../../lib/model_interface.mjs';
import assert from 'assert';
const expect = chai.expect;
const array = {
  reshape: TensorScriptModelInterface.reshape,
  flatten,
};
const reshape = array.reshape;
describe('util.array', function () {
  /** @test {../../lib/model_interface.mjs~size} */
  describe('size', function () {
    it('should calculate the size of a scalar', function () {
      assert.deepEqual(size(2), []);
      assert.deepEqual(size('string'), []);
    });

    it('should calculate the size of a 1-dimensional array', function () {
      assert.deepEqual(size([]), [0, ]);
      assert.deepEqual(size([1, ]), [1, ]);
      assert.deepEqual(size([1, 2, 3, ]), [3, ]);
    });

    it('should calculate the size of a 2-dimensional array', function () {
      assert.deepEqual(size([[], ]), [1, 0, ]);
      assert.deepEqual(size([[], [], ]), [2, 0, ]);
      assert.deepEqual(size([[1, 2, ], [3, 4, ], ]), [2, 2, ]);
      assert.deepEqual(size([[1, 2, 3, ], [4, 5, 6, ], ]), [2, 3, ]);
    });

    it('should calculate the size of a 3-dimensional array', function () {
      assert.deepEqual(size([[[], ], ]), [1, 1, 0, ]);
      assert.deepEqual(size([[[], [], ], ]), [1, 2, 0, ]);
      assert.deepEqual(size([[[], [], ], [[], [], ], ]), [2, 2, 0, ]);
      assert.deepEqual(size([[[1, ], [2, ], ], [[3, ], [4, ], ], ]), [2, 2, 1, ]);
      assert.deepEqual(size([[[1, 2, ], [3, 4, ], ], [[5, 6, ], [7, 8, ], ], ]), [2, 2, 2, ]);
      assert.deepEqual(size([
        [[1, 2, 3, 4, ], [5, 6, 7, 8, ], ],
        [[1, 2, 3, 4, ], [5, 6, 7, 8, ], ],
        [[1, 2, 3, 4, ], [5, 6, 7, 8, ], ],
      ]), [3, 2, 4, ]);
    });

    it('should not validate whether all dimensions match', function () {
      assert.deepEqual(size([[1, 2, ], [3, 4, 5, ], ]), [2, 2, ]);
    });
  });
  /** @test {../../lib/model_interface.mjs~reshape} */
  describe('reshape', function () {
    it('should reshape a 1 dimensional array into a 2 dimensional array', function () {
      const a = [1, 2, 3, 4, 5, 6, 7, 8, ];

      assert.deepEqual(
        reshape(a, [2, 4, ]),
        [[1, 2, 3, 4, ],
          [5, 6, 7, 8, ], ]
      );
      assert.deepEqual(
        reshape(a, [4, 2, ]),
        [[1, 2, ],
          [3, 4, ],
          [5, 6, ],
          [7, 8, ], ]
      );
      assert.deepEqual(
        reshape(a, [1, 8, ]),
        [[1, 2, 3, 4, 5, 6, 7, 8, ], ]
      );
      assert.deepEqual(
        reshape(a, [1, 1, 8, ]),
        [[[1, 2, 3, 4, 5, 6, 7, 8, ], ], ]
      );
    });

    it('should reshape a 2 dimensional array into a 1 dimensional array', function () {
      const a = [
        [0, 1, ],
        [2, 3, ],
      ];

      assert.deepEqual(
        reshape(a, [4, ]),
        [0, 1, 2, 3, ]
      );
    });

    it('should reshape a 3 dimensional array', function () {
      const a = [[[1, 2, ],
        [3, 4, ], ],

      [[5, 6, ],
        [7, 8, ], ], ];

      assert.deepEqual(
        reshape(a, [8, ]),
        [1, 2, 3, 4, 5, 6, 7, 8, ]
      );

      assert.deepEqual(
        reshape(a, [2, 4, ]),
        [[1, 2, 3, 4, ],
          [5, 6, 7, 8, ], ]
      );
    });

    it('should throw an error when reshaping to a dimension with length 0', function () {
      assert.throws(function () {
        reshape([1, 2, ], [0, 2, ]); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape([1, 2, ], [2, 0, ]); 
      }, /DimensionError/);
    });

    it('should throw an error when reshaping a non-empty array to an empty array', function () {
      assert.throws(function () {
        reshape([1, ], []); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape([1, 2, ], []); 
      }, /DimensionError/);
    });

    it('should throw an error when reshaping to a size that differs from the original', function () {
      const a = [1, 2, 3, 4, 5, 6, 7, 8, 9, ];

      assert.deepEqual(
        reshape(a, [3, 3, ]),
        [[1, 2, 3, ],
          [4, 5, 6, ],
          [7, 8, 9, ], ]
      );
      assert.throws(function () {
        reshape(a, [3, 2, ]); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape(a, [2, 3, ]); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape(a, [3, 3, 3, ]); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape(a, [3, 4, ]); 
      }, /DimensionError/);
      assert.throws(function () {
        reshape(a, [4, 3, ]); 
      }, /DimensionError/);
    });

    it('should throw an error in case of wrong type of arguments', function () {
      assert.throws(function () {
        reshape([], 2); 
      }, /Array expected/);
      assert.throws(function () {
        reshape(2); 
      }, /Array expected/);
    });
  });
  /** @test {../../lib/model_interface.mjs~flatten} */
  describe('flatten', function () {
    it('should flatten a scalar', function () {
      assert.deepEqual(array.flatten(1), 1);
    });

    it('should flatten a 1 dimensional array', function () {
      assert.deepEqual(array.flatten([1, 2, 3, ]), [1, 2, 3, ]);
    });

    it('should flatten a 2 dimensional array', function () {
      assert.deepEqual(array.flatten([[1, 2, ], [3, 4, ], ]), [1, 2, 3, 4, ]);
    });

    it('should flatten a 3 dimensional array', function () {
      assert.deepEqual(array.flatten([[[1, 2, ], [3, 4, ], ], [[5, 6, ], [7, 8, ], ], ]), [1, 2, 3, 4, 5, 6, 7, 8, ]);
    });

    it('should return a new array', function () {
      const input = [3, 2, 1, ];
      const flat = array.flatten(input);
      flat.sort();
      assert.deepEqual(input, [3, 2, 1, ]);
    });
  });
});