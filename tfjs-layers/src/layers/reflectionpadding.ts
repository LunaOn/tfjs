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

import { InputSpec, Layer, LayerArgs } from '../engine/topology';
import { ValueError } from '../errors';
import { DataFormat, Shape } from '../keras_format/common';
import { Kwargs } from '../types';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';

/**
 * Pads the middle dimension of a 3D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of 2 integers, how many zeros to add at the start and
 *   end of the middle dimension (i.e., dimension 1).
 * @return A padded 3D `tf.Tensor`.
 */
export function temporalPadding(x: Tensor, padding?: [number, number]): Tensor {
  return tidy(() => {
    if (x.rank !== 3) {
      throw new ValueError(
        `temporalPadding expects input tensor to be 3-D, but received a ` +
        `${x.rank}-D tensor.`);
    }

    if (padding == null) {
      padding = [1, 1];
    }
    if (padding.length !== 2) {
      throw new ValueError(
        `temporalPadding expects input padding pattern to be a length-2 ` +
        `array, but received a length-${padding.length} array.`);
    }

    const pattern: Array<[number, number]> = [[0, 0], padding, [0, 0]];
    return tfc.pad(x, pattern);
  });
}

/**
 * Pads the 2nd and 3rd dimensions of a 4D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of two `Array`s, each of which is an `Array` of two
 *   integers. The amount of padding at the beginning and end of the 2nd and 3rd
 *   dimensions, respectively.
 * @param dim_ordering 'channelsLast' (default) or 'channelsFirst'.
 * @return Padded 4D `tf.Tensor`.
 */
export function reflection2dPadding(
  x: Tensor, padding?: [[number, number], [number, number]],
  dim_ordering?: DataFormat): Tensor {
  const paddingNum = padding[0][0]
  const width = x.shape[2]
  const height = x.shape[1]
  // 右边的切片index
  let rightIndex = []
  for (let i = 1; i <= paddingNum; i++) {
    rightIndex.push(width - i)
  }
  // 左边和上边的切片index
  let leftTopIndex = []
  for (let i = paddingNum - 1; i >= 0; i--) {
    leftTopIndex.push(i)
  }

  // 右侧
  let indices = tfc.tensor1d(rightIndex, 'int32')
  let b = x.gather(indices, 2)  // 切片
  let result = tfc.concat([x, b], 2) // 连接

  // 左侧
  indices = tfc.tensor1d(leftTopIndex, 'int32')
  b = x.gather(indices, 2)  // 切片
  result = tfc.concat([b, result], 2) // 连接

  // 上边
  indices = tfc.tensor1d(leftTopIndex, 'int32')
  b = result.gather(indices, 1)  // 切片
  result = tfc.concat([b, result], 1) // 连接

  // 下边
  let bottomIndex = []
  for (let i = 1; i <= paddingNum; i++) {
    bottomIndex.push(height + paddingNum - i)
  }
  indices = tfc.tensor1d(bottomIndex, 'int32')
  b = result.gather(indices, 1)  // 切片
  result = tfc.concat([result, b], 1) // 连接

  return result
}

export declare interface ReflectionPadding2DLayerArgs extends LayerArgs {
  /**
   * Integer, or `Array` of 2 integers, or `Array` of 2 `Array`s, each of
   * which is an `Array` of 2 integers.
   * - If integer, the same symmetric padding is applied to width and height.
   * - If Array` of 2 integers, interpreted as two different symmetric values
   *   for height and width:
   *   `[symmetricHeightPad, symmetricWidthPad]`.
   * - If `Array` of 2 `Array`s, interpreted as:
   *   `[[topPad, bottomPad], [leftPad, rightPad]]`.
   */
  padding?: number | [number, number] | [[number, number], [number, number]];

  /**
   * One of `'channelsLast'` (default) and `'channelsFirst'`.
   *
   * The ordering of the dimensions in the inputs.
   * `channelsLast` corresponds to inputs with shape
   * `[batch, height, width, channels]` while `channelsFirst`
   * corresponds to inputs with shape
   * `[batch, channels, height, width]`.
   */
  dim_ordering?: string;
}

export class ReflectionPadding2D extends Layer {
  /** @nocollapse */
  static className = 'ReflectionPadding2D';
  readonly dim_ordering: string;
  readonly padding: [[number, number], [number, number]];

  constructor(args?: ReflectionPadding2DLayerArgs) {
    if (args == null) {
      args = {};
    }
    super(args);

    this.dim_ordering =
      args.dim_ordering == null ? 'default' : args.dim_ordering;
    // TODO(cais): Maybe refactor the following logic surrounding `padding`
    //   into a helper method.
    if (args.padding == null) {
      this.padding = [[1, 1], [1, 1]];
    } else if (typeof args.padding === 'number') {
      this.padding =
        [[args.padding, args.padding], [args.padding, args.padding]];
    } else {
      args.padding = args.padding;
      if (args.padding.length !== 2) {
        throw new ValueError(
          `ReflectionPadding2D expects padding to be a length-2 array, but ` +
          `received a length-${args.padding.length} array.`);
      }

      let heightPadding: [number, number];
      let widthPadding: [number, number];
      if (typeof args.padding[0] === 'number') {
        heightPadding = [args.padding[0], args.padding[0]];
        widthPadding = [args.padding[1] as number, args.padding[1] as number];
      } else {
        args.padding = args.padding as [[number, number], [number, number]];

        if (args.padding[0].length !== 2) {
          throw new ValueError(
            `ReflectionPadding2D expects height padding to be a length-2 array, ` +
            `but received a length-${args.padding[0].length} array.`);
        }
        heightPadding = args.padding[0] as [number, number];

        if (args.padding[1].length !== 2) {
          throw new ValueError(
            `ReflectionPadding2D expects width padding to be a length-2 array, ` +
            `but received a length-${args.padding[1].length} array.`);
        }
        widthPadding = args.padding[1] as [number, number];
      }
      this.padding = [heightPadding, widthPadding];
    }
    this.inputSpec = [new InputSpec({ ndim: 4 })];
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    inputShape = getExactlyOneShape(inputShape);

    let rows: number;
    let cols: number;
    if (inputShape[1] != null && inputShape[1] >= 0) {
      rows = inputShape[1] + this.padding[0][0] + this.padding[0][1];
    } else {
      rows = null;
    }
    if (inputShape[2] != null && inputShape[2] >= 0) {
      cols = inputShape[2] + this.padding[1][0] + this.padding[1][1];
    } else {
      cols = null;
    }
    return [inputShape[0], rows, cols, inputShape[3]];
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(
      () => reflection2dPadding(
        getExactlyOneTensor(inputs), this.padding));
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      padding: this.padding
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(ReflectionPadding2D);
