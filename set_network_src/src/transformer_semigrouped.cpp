#include "network_builders.h"

NetworkProfile BuildTransformerSemiGrouped() {
  NetworkProfile net{"transformer_semigrouped", 128, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"embed", "op", 1.8000e9, 39.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_sg_1", {
      LayerSpec{"grouped_attn1", "op", 4.2000e9, 42.0000},
      LayerSpec{"mlp1", "op", 6.7000e9, 44.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_sg_2", {
      LayerSpec{"grouped_attn2", "op", 4.1000e9, 42.0000},
      LayerSpec{"mlp2", "op", 6.6000e9, 44.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"proj", "op", 2.0000e9, 14.0000},
  }});
  return net;
}
