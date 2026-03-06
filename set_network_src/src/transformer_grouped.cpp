#include "network_builders.h"

NetworkProfile BuildTransformerGrouped() {
  NetworkProfile net{"transformer_grouped", 128, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"embed", "op", 1.8000e9, 39.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_group_1", {
      LayerSpec{"attn_g1", "op", 4.6000e9, 44.0000},
      LayerSpec{"mlp_g1", "op", 6.9000e9, 45.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_group_2", {
      LayerSpec{"attn_g2", "op", 4.5000e9, 44.0000},
      LayerSpec{"mlp_g2", "op", 6.8000e9, 45.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"proj", "op", 2.0000e9, 14.0000},
  }});
  return net;
}
