import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
import alias from 'rollup-plugin-alias';
// import async from 'rollup-plugin-async';
import builtins from 'rollup-plugin-node-builtins';
import globals from 'rollup-plugin-node-globals';
import pkg from './package.json';

export default [
  // browser-friendly UMD build
  {
    input: 'index.mjs',
    output: {
      exports: 'named',
      file: pkg.browser,
      name: 'tensorscript',
      format: 'umd',
    },
    plugins: [
      resolve({
        preferBuiltins: true,
      }), // so Rollup can find `ms`
      commonjs({
        namedExports: {
          // left-hand side can be an absolute path, a path
          // relative to the current directory, or the name
          // of a module in node_modules
          'node_modules/@tensorflow/tfjs/dist/tf.esm.js': [ 'tf', ],
        },
      }), // so Rollup can convert `ms` to an ES module
      builtins({
      }),
      globals({
      }),
    ],
  },

  // CommonJS (for Node) and ES module (for bundlers) build.
  // (We could have three entries in the configuration array
  // instead of two, but it's quicker to generate multiple
  // builds from a single configuration where possible, using
  // an array for the `output` option, where we can specify 
  // `file` and `format` for each target)
  {
    input: 'index.mjs',
    external: [
      '@tensorflow/tfjs',
      'lodash.range',
      // 'lodash.rangeright'
    ],
    output: [
      {
        exports: 'named',
        file: pkg.node,
        name: 'tensorscript',
        format: 'cjs',
      },
      {
        exports: 'named',
        file: pkg.es,
        name: 'tensorscript',
        format: 'es',
      },
    ],
  },
];
