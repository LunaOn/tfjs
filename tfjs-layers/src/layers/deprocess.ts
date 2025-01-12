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

import { Shape } from '../keras_format/common';
import { Layer, LayerArgs } from '../engine/topology';

import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';

export declare interface DeprocessStylizedImageLayerArgs extends LayerArgs {
  activation?: string;
}

export class DeprocessStylizedImage extends Layer {
  /** @nocollapse */
  static className = 'DeprocessStylizedImage';
  readonly activation: string;
  constructor(args?: DeprocessStylizedImageLayerArgs) {
    if (args == null) {
      args = {};
    }
    super(args);
    this.activation = args.activation
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    return inputShape;
  }

  call(inputs: Tensor | Tensor[]): Tensor | Tensor[] {
    let x = getExactlyOneTensor(inputs)
    return tidy(() => {
      if (this.activation === 'tanh') {
        x = tfc.mul(tfc.add(x, 1.0), 127.5)
      } else {
        x = tfc.mul(x, 255)
      }
      return x
    })
  }

  getConfig(): serialization.ConfigDict {
    return super.getConfig();
  }
}
serialization.registerClass(DeprocessStylizedImage);
