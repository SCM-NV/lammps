/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(amspipe,FixAMSPipe);
// clang-format on
#else

#ifndef LMP_FIX_AMSPIPE_H
#define LMP_FIX_AMSPIPE_H

#include "fix.h"

#include "amspipe.hpp"

#include <set>
#include <vector>

namespace LAMMPS_NS {

class FixAMSPipe : public Fix {
 public:
  FixAMSPipe(class LAMMPS *, int, char **);
  ~FixAMSPipe() override;
  int setmask() override;
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void pre_exchange() override;

 private:
  class Irregular *irregular;
  AMSPipe *pipe;

  bool exiting;

  // Variables holding our current system:
  std::vector<std::string> atomSymbols;
  std::vector<double>      coords;
  std::vector<double>      latticeVectors;
  double                   totalCharge = 0.0;
  std::vector<int64_t>     bonds;
  std::vector<double>      bondOrders;
  std::vector<std::string> atomicInfo;

  std::vector<double>      gradients;

  // Cache of results we have kept:
  std::set<std::string> keptResults; // For this demo we just keep their titles and no actual data ...

  std::vector<double>      prevFrac;

  double gradconv, potconv, posconv, stressconv, posconv3;

};

}    // namespace LAMMPS_NS

#endif
#endif
