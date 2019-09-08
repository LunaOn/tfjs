/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Padding Layers.
 */

// Porting Note: In Python Keras, the padding layers are in convolutional.py,
//   but we decided to put them in a separate file (padding.ts) for clarity.

import * as tfc from '@tensorflow/tfjs-core';
import { serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
import { ValueError } from '../errors';

import { Shape } from '../keras_format/common';
import { InputSpec } from '../engine/topology';
import * as initializers from '../initializers';
import * as constraints from '../constraints';
import * as regularizers from '../regularizers';
import { Layer, LayerArgs } from '../engine/topology';
import { LayerVariable } from '../variables';
import { getExactlyOneTensor } from '../utils/types_utils';

import { Kwargs } from '../types';
import { getExactlyOneShape } from '../utils/types_utils';
import * as math_utils from '../utils/math_utils';
import * as generic_utils from '../utils/generic_utils';
// import { normalizeBatchInTraining } from './normalization';

export declare interface ConditionalInstanceNormalizationLayerArgs extends LayerArgs {
  axis?: number;
  epsilon?: number;
  center?: boolean;
  scale?: boolean;
  betaInitializer?: string;
  gammaInitializer?: string;
  betaRegularizer?: string;
  gammaRegularizer?: string;
  betaConstraint?: string;
  gammaConstraint?: string;
  styleNum?: number;
}

export class ConditionalInstanceNormalization extends Layer {
  /** @nocollapse */
  static className = 'ConditionalInstanceNormalization';
  readonly axis?: number;
  readonly epsilon?: number;
  readonly center?: boolean;
  readonly scale?: boolean;
  readonly betaInitializer?: initializers.Initializer;
  readonly gammaInitializer?: initializers.Initializer;
  readonly betaRegularizer?: regularizers.Regularizer;
  readonly gammaRegularizer?: regularizers.Regularizer;
  readonly betaConstraint?: constraints.Constraint;
  readonly gammaConstraint?: constraints.Constraint;

  readonly style_num: number;

  private gamma: LayerVariable;
  private beta: LayerVariable;

  private sw?: number[];

  constructor(args?: ConditionalInstanceNormalizationLayerArgs) {
    if (args == null) {
      args = {};
    }
    super(args);

    this.axis = args.axis == null ? -1 : args.axis;
    this.epsilon = args.epsilon == null ? 1e-3 : args.epsilon;
    this.center = args.center;
    this.scale = args.scale;
    this.style_num = args.styleNum;

    this.betaInitializer = initializers.getInitializer("zeros")
    this.gammaInitializer = initializers.getInitializer("ones")
    this.betaRegularizer = regularizers.getRegularizer(args.betaRegularizer)
    this.gammaRegularizer = regularizers.getRegularizer(args.gammaRegularizer)
    this.betaConstraint = constraints.getConstraint(args.betaConstraint)
    this.gammaConstraint = constraints.getConstraint(args.gammaConstraint)
  }

  public build(inputShape: Shape | Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
    const axis = this.axis >= 0 ? this.axis : (this.axis + inputShape.length);
    const dim = inputShape[axis];
    if (dim == null) {
      throw new ValueError(
        `Axis ${axis} of input tensor should have a defined dimension but ` +
        `the layer received an input with shape ` +
        `${JSON.stringify(inputShape)}.`);
    }
    this.inputSpec =
      [new InputSpec({ ndim: inputShape.length, axes: { [axis]: dim } })];
    const shape = [dim];

    this.sw = [];
    for (let i = 0; i < this.style_num; i++) {
      this.sw.push(0);
    }
    this.sw[0] = 1;

    if (this.sw != null) {
      const new_shape = [this.style_num, dim];
      this.gamma = this.addWeight(
        'gamma', new_shape, null, this.gammaInitializer, this.gammaRegularizer,
        true, this.gammaConstraint);

      this.beta = this.addWeight(
        'beta', new_shape, null, this.betaInitializer, this.betaRegularizer, true,
        this.betaConstraint);
    } else {
      if (this.scale) {
        this.gamma = this.addWeight(
          'gamma', shape, null, this.gammaInitializer, this.gammaRegularizer,
          true, this.gammaConstraint);
      } else {
        this.gamma = null;
      }

      if (this.center) {
        this.beta = this.addWeight(
          'beta', shape, null, this.betaInitializer, this.betaRegularizer, true,
          this.betaConstraint);
      } else {
        this.beta = null;
      }
    }

    this.built = true;
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    // inputShape = getExactlyOneShape(inputShape);
    return inputShape;
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      const input = getExactlyOneTensor(inputs);
      const inputShape = input.shape;
      const ndim = inputShape.length;
      const reductionAxes = math_utils.range(0, ndim);
      const axis = this.axis >= 0 ? this.axis : (this.axis + ndim);
      reductionAxes.splice(axis, 1);
      const broadcastShape = generic_utils.pyListRepeat(1, ndim);
      broadcastShape[axis] = inputShape[axis];

      const sortedReductionAxes = reductionAxes.slice();
      sortedReductionAxes.sort();


      // const [normed,] = normalizeBatchInTraining(input, broadcastGamma, broadcastBeta, reductionAxes, this.epsilon);

      const meanAndVar = tfc.moments(input, reductionAxes);
      const mean = meanAndVar.mean;
      const variance = meanAndVar.variance;
      const stddev = tfc.sqrt(variance).add(this.epsilon);

      var normed = input.sub(mean).div(stddev);

      if (this.sw != null) {
        const betaW = tfc.dot(this.sw, this.beta.read());
        const gammaW = tfc.dot(this.sw, this.gamma.read());

        const broadcastBeta = tfc.reshape(betaW, broadcastShape);
        const broadcastGamma = tfc.reshape(gammaW, broadcastShape);
        normed = normed.mul(broadcastGamma);
        normed = normed.add(broadcastBeta);
      } else {
        const broadcastBeta =
          this.center ? this.beta.read().reshape(broadcastShape) : null;
        const broadcastGamma =
          this.scale ? this.gamma.read().reshape(broadcastShape) : null;

        if (this.scale) {
          normed = normed.mul(broadcastGamma);
        }
        if (this.center) {
          normed = normed.add(broadcastBeta);
        }
      }

      return normed;
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      axis: this.axis,
      epsilon: this.epsilon.valueOf(),
      center: this.center,
      scale: this.scale,
      beta_initializer: initializers.serializeInitializer(this.betaInitializer),
      gamma_initializer: initializers.serializeInitializer(this.gammaInitializer),
      beta_regularizer: regularizers.serializeRegularizer(this.betaRegularizer),
      gamma_regularizer: regularizers.serializeRegularizer(this.gammaRegularizer),
      beta_constraint: constraints.serializeConstraint(this.betaConstraint),
      gamma_constraint: constraints.serializeConstraint(this.gammaConstraint),
      style_num: this.style_num
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ConditionalInstanceNormalization);
