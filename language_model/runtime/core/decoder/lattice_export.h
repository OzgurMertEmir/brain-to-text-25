// decoder/lattice_export.h
#ifndef DECODER_LATTICE_EXPORT_H_
#define DECODER_LATTICE_EXPORT_H_

#include <tuple>
#include <vector>

#include "kaldi/lat/kaldi-lattice.h"

namespace wenet {

// Each arc: (src_state, dst_state, ilabel, olabel, graph_cost, acoustic_cost)
using ArcTuple   = std::tuple<int, int, int, int, float, float>;
// Each final state: (state, graph_cost, acoustic_cost)
using FinalTuple = std::tuple<int, float, float>;

struct LatticeExport {
  int start_state;
  std::vector<ArcTuple> arcs;
  std::vector<FinalTuple> finals;
};

// Convert a kaldi::Lattice into a plain-struct representation
LatticeExport ExportLattice(const kaldi::Lattice& lat);

}  // namespace wenet

#endif  // DECODER_LATTICE_EXPORT_H_
