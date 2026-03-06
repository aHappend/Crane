#include "network_builders.h"

NetworkProfile BuildLLM() {
  NetworkProfile net{"llm", 128, {}};
  net.blocks.push_back(BlockSpec{"embedding", {
      LayerSpec{"token_embed", "op", 2.5000e9, 50.0000},
      LayerSpec{"rope", "op", 0.4000e9, 50.0000},
  }});
  net.blocks.push_back(BlockSpec{"decoder_stack_1", {
      LayerSpec{"attn1", "op", 8.0000e9, 56.0000},
      LayerSpec{"mlp1", "op", 12.5000e9, 58.0000},
  }});
  net.blocks.push_back(BlockSpec{"decoder_stack_2", {
      LayerSpec{"attn2", "op", 7.8000e9, 56.0000},
      LayerSpec{"mlp2", "op", 12.0000e9, 58.0000},
  }});
  net.blocks.push_back(BlockSpec{"decoder_stack_3", {
      LayerSpec{"attn3", "op", 7.6000e9, 55.0000},
      LayerSpec{"mlp3", "op", 11.8000e9, 57.0000},
  }});
  net.blocks.push_back(BlockSpec{"lm_head", {
      LayerSpec{"norm", "op", 0.6000e9, 30.0000},
      LayerSpec{"head", "op", 2.0000e9, 12.0000},
  }});
  return net;
}
