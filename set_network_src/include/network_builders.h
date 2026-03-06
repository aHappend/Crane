#pragma once

#include "network_profile.h"

NetworkProfile BuildAlexNet();
NetworkProfile BuildAttentionGroupBuilder();
NetworkProfile BuildDarkNet19();
NetworkProfile BuildDenseNet();
NetworkProfile BuildGNMT();
NetworkProfile BuildGoogleNet();
NetworkProfile BuildIncepResNet();
NetworkProfile BuildLLM();
NetworkProfile BuildPNASNet();
NetworkProfile BuildResNet();
NetworkProfile BuildTransformerFused();
NetworkProfile BuildTransformerGrouped();
NetworkProfile BuildTransformerSemiFused();
NetworkProfile BuildTransformerSemiGrouped();
NetworkProfile BuildTransformer();
NetworkProfile BuildVGG();
NetworkProfile BuildZFNet();

std::vector<NetworkProfile> BuildAllNetworks();
