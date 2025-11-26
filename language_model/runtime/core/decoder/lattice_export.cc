// decoder/lattice_export.cc
#include "decoder/lattice_export.h"
#include "fst/fstlib.h"

namespace wenet {

LatticeExport ExportLattice(const kaldi::Lattice& lat) {
  LatticeExport out;
  out.start_state = lat.Start();

  // Iterate over states
  for (fst::StateIterator<kaldi::Lattice> siter(lat);
       !siter.Done(); siter.Next()) {
    int s = siter.Value();

    // Final weights (if any)
    kaldi::LatticeWeight f = lat.Final(s);
    if (f != kaldi::LatticeWeight::Zero()) {
      out.finals.emplace_back(
          s,
          f.Value1(),  // graph / LM cost
          f.Value2()   // acoustic cost
      );
    }

    // Outgoing arcs
    for (fst::ArcIterator<kaldi::Lattice> aiter(lat, s);
         !aiter.Done(); aiter.Next()) {
      const kaldi::LatticeArc& arc = aiter.Value();
      out.arcs.emplace_back(
          s,                   // src
          arc.nextstate,       // dst
          arc.ilabel,          // input label
          arc.olabel,          // output label
          arc.weight.Value1(), // graph / LM cost
          arc.weight.Value2()  // acoustic cost
      );
    }
  }

  return out;
}

}  // namespace wenet