#include "network_builders.h"

NetworkProfile BuildTransformerSemiFused() {
  NetworkProfile net{"transformer_semifused", 128, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"embed", "op", 1.8000e9, 39.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_sf_1", {
      LayerSpec{"attn1", "op", 4.4000e9, 43.0000},
      LayerSpec{"fused_mlp1", "op", 7.0000e9, 45.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_sf_2", {
      LayerSpec{"attn2", "op", 4.3000e9, 43.0000},
      LayerSpec{"fused_mlp2", "op", 6.9000e9, 45.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"proj", "op", 2.0000e9, 14.0000},
  }});
  return net;
}
