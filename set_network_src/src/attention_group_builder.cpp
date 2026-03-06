#include "network_builders.h"

NetworkProfile BuildAttentionGroupBuilder() {
  NetworkProfile net{"attention_group_builder", 96, {}};
  net.blocks.push_back(BlockSpec{"embed", {
      LayerSpec{"token_embed", "op", 1.2000e9, 36.0000},
      LayerSpec{"pos_embed", "op", 0.3000e9, 36.0000},
  }});
  net.blocks.push_back(BlockSpec{"attn_group_a", {
      LayerSpec{"qkv_proj_a", "op", 2.2000e9, 34.0000},
      LayerSpec{"mh_attn_a", "op", 3.4000e9, 38.0000},
  }});
  net.blocks.push_back(BlockSpec{"attn_group_b", {
      LayerSpec{"qkv_proj_b", "op", 2.2000e9, 34.0000},
      LayerSpec{"mh_attn_b", "op", 3.4000e9, 38.0000},
  }});
  net.blocks.push_back(BlockSpec{"mlp_group", {
      LayerSpec{"mlp_up", "op", 4.0000e9, 40.0000},
      LayerSpec{"mlp_down", "op", 3.8000e9, 36.0000},
  }});
  return net;
}
