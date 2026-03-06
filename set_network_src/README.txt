SET-compatible network src bundle

This directory provides approximate network profiles for scheduling experiments.
It matches the file naming style:
  alexnet.cpp, attention_group_builder.cpp, darknt19.cpp, densenet.cpp,
  gnmt.cpp, googlenet.cpp, incep_resnet.cpp, llm.cpp, pnasnet.cpp, resnet.cpp,
  transformer_fused.cpp, transformer_grouped.cpp, transformer_semifused.cpp,
  transformer_semigrouped.cpp, transformer.cpp, vgg.cpp, zfnet.cpp

Notes:
1) These are scheduling-oriented proxy profiles, not exact vendor training code.
2) Each network has a JSON twin in profiles/*.json for direct Python-side experiments.
3) You can copy src/*.cpp + include/*.h into your SET codebase and map LayerSpec fields.
