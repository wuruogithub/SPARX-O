# SPARX-O + U-Net + PaPs

This directory provides the attributed Exchanger4SITS/PaPs data, training,
test and metric framework, with SPARX-O replacing the temporal encoder.
See the root README for installation, fold mapping, temporal initialization
and training commands. The semantic boundary-affinity refinement head is not
used by this alternative panoptic decoder.

The files retain historical experiment names for checkpoint compatibility.
The Exchanger4SITS MIT license is in `LICENSE.exchanger4sits`.
The PaPs head credits Vivien Sainte Fare Garnot in its source header.
Supporting upstream modules are retained to preserve the original framework's
imports; only the SPARX-O + U-Net + PaPs path is part of the main release.
