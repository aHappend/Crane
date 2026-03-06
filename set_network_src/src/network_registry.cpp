#include "network_builders.h"

std::vector<NetworkProfile> BuildAllNetworks() {
  return {
      BuildAlexNet(),
      BuildAttentionGroupBuilder(),
      BuildDarkNet19(),
      BuildDenseNet(),
      BuildGNMT(),
      BuildGoogleNet(),
      BuildIncepResNet(),
      BuildLLM(),
      BuildPNASNet(),
      BuildResNet(),
      BuildTransformerFused(),
      BuildTransformerGrouped(),
      BuildTransformerSemiFused(),
      BuildTransformerSemiGrouped(),
      BuildTransformer(),
      BuildVGG(),
      BuildZFNet(),
  };
}
