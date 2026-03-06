#include "network_builders.h"

NetworkProfile BuildTransformerFused() {
  NetworkProfile net{"transformer_fused", 128, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"embed", "op", 1.9000e9, 40.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_fused_1", {
      LayerSpec{"fused_attn_mlp1", "op", 9.5000e9, 46.0000},
  }});
  net.blocks.push_back(BlockSpec{"enc_fused_2", {
      LayerSpec{"fused_attn_mlp2", "op", 9.1000e9, 46.0000},
  }});
  net.blocks.push_back(BlockSpec{"head", {
      LayerSpec{"proj", "op", 2.1000e9, 14.0000},
  }});
  return net;
}
