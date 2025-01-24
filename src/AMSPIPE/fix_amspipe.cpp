/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_amspipe.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "irregular.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "timer.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

static AMSPipe *saved_pipe = nullptr;

/******************************************************************************************
 * This fix enables LAMMPS to serve as an AMSPipe worker.
 ******************************************************************************************/

FixAMSPipe::FixAMSPipe(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), irregular(new Irregular(lmp)), pipe(nullptr), exiting(false)
{
  if (strcmp(style, "amspipe") != 0 && narg != 3) error->all(FLERR, "Illegal fix amspipe command");

  if (strcmp(arg[1], "all") != 0) error->warning(FLERR, "Fix amspipe always uses group all");

  if (atom->tag_enable == 0) error->all(FLERR, "Cannot use fix amspipe without atom IDs");

  if (atom->tag_consecutive() == 0) error->all(FLERR, "Fix amspipe requires consecutive atom IDs");

  modify->add_compute("amspipe_pressure all pressure NULL virial");

  // conversions from LAMMPS "real" units to atomic units, which are used by AMSPipe
  // empirical reconstruction of the conversion through ASE
  potconv = 3.166815455e-06 / force->boltz;
  // conversion using force->boltz from LAMMPS "electron" units
  // potconv=3.16681534e-06/force->boltz;
  posconv = 0.5291772105638411 * force->angstrom;
  posconv3 = posconv * posconv * posconv;
  gradconv = -1 * potconv * posconv;
  stressconv = -1 / force->nktv2p * potconv * posconv3;
}

/* ---------------------------------------------------------------------- */

FixAMSPipe::~FixAMSPipe()
{
  delete irregular;
  delete pipe;
}

/* ---------------------------------------------------------------------- */

int FixAMSPipe::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAMSPipe::init()
{
  if (comm->me == 0) {
    if (saved_pipe) {
      pipe = saved_pipe;
      saved_pipe = nullptr;
    } else {
      pipe = new AMSPipe();
    }
  }

  // asks for evaluation of PE at first step
  modify->compute[modify->find_compute("thermo_pe")]->invoked_scalar = -1;
  modify->addstep_compute_all(update->ntimestep + 1);
}

void FixAMSPipe::initial_integrate(int /*vflag*/)
{
  // Variable to store the error until we send the corresponding return message:
  std::unique_ptr<AMSPipe::Error> error;
  bool lattice_changed = false;

  while (true) {
    auto msg = pipe->receive();

    try {
      if (msg.name == "Exit") {
        update->nsteps = update->ntimestep - 1;
        exiting = true;

        break;

      } else if (error) {    // We still have an error buffered
        if (msg.name.rfind("Set", 0) == 0) {
          // Calls to "Set" methods are ignored while an error is buffered.
          continue;
        } else {
          // Non-"Set" method called: return buffered error and clear it.
          pipe->send_return(error->status, error->method, error->argument, error->what());
          error.reset();
          if (exiting) {
            saved_pipe = pipe;
            pipe = nullptr;
            return;
          }
        }

      } else if (msg.name == "Hello") {
        int64_t version;
        pipe->extract_Hello(msg, version);
        pipe->send_return(version == 1 ? AMSPipe::Status::success
                                       : AMSPipe::Status::unknown_version);

      } else if (msg.name == "SetCoords") {
        pipe->extract_SetCoords(msg, coords.data());

      } else if (msg.name == "SetLattice") {
        int prevLatticeSize = latticeVectors.size();
        pipe->extract_SetLattice(msg, latticeVectors);

        if (latticeVectors.size() != prevLatticeSize) {
          update->nsteps = update->ntimestep - 1;
          exiting = true;
          throw AMSPipe::Error(AMSPipe::Status::runtime_error, msg.name, "",
                               "Reinitialization required");
        }

        lattice_changed = true;

      } else if (msg.name == "SetSystem") {
        std::vector<std::string> prevAtomSymbols = std::move(atomSymbols);
        int prevLatticeSize = latticeVectors.size();
        pipe->extract_SetSystem(msg, atomSymbols, coords, latticeVectors, totalCharge, bonds,
                                bondOrders, atomicInfo);

        if (!prevAtomSymbols.empty() &&
            (atomSymbols != prevAtomSymbols || latticeVectors.size() != prevLatticeSize)) {
          update->nsteps = update->ntimestep - 1;
          exiting = true;
          throw AMSPipe::Error(AMSPipe::Status::runtime_error, msg.name, "",
                               "Reinitialization required");
        }

        lattice_changed = true;

      } else if (msg.name == "Solve") {
        AMSPipe::SolveRequest request;
        bool keepResults;
        std::string prevTitle;

        pipe->extract_Solve(msg, request, keepResults, prevTitle);

        if (keptResults.find(request.title) != keptResults.end()) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "Solve", "title",
                               "title in request corresponds to an already stored results object");
        }
        if (!prevTitle.empty() && keptResults.find(prevTitle) == keptResults.end()) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "Solve", "prevTitle",
                               "prevTitle does not correspond to a kept results object");
        }
        if (keepResults) keptResults.insert(request.title);

        int nVectors = latticeVectors.size() / 3;

        if (lattice_changed) {
          for (int d = 0; d < nVectors; d++) {
            domain->boxlo[d] = 0;
            domain->boxhi[d] = latticeVectors[3 * d + d] * posconv;
          }

          // do error checks on simulation box and set small for triclinic boxes
          domain->set_initial_box();
          // reset global and local box using the new box dimensions
          domain->reset_box();
          // signal that the box has (or may have) changed
          domain->box_change = 1;

          lattice_changed = true;
        }

        if (nVectors > 0) {
          std::vector<double> fractional(coords.size());
          for (int i = 0; i < coords.size(); i += 3) {
            for (int d = 0; d < nVectors; d++) {
              fractional[i + d] = coords[i + d] / latticeVectors[3 * d + d];
            }
          }
          std::vector<int> shift(coords.size());
          if (prevFrac.size() == coords.size()) {
            for (int i = 0; i < coords.size(); i++) {
              shift[i] = std::nearbyint(fractional[i] - prevFrac[i]);
            }
          } else {
            prevFrac.clear();
            for (int i = 0; i < coords.size(); i++) { shift[i] = std::floor(fractional[i]); }
          }
          for (int i = 0; i < coords.size(); i++) { fractional[i] -= shift[i]; }
          for (int i = 0; i < coords.size(); i += 3) {
            for (int d = 0; d < nVectors; d++) {
              coords[i + d] = fractional[i + d] * latticeVectors[3 * d + d];
            }
          }
        }

        double **x = atom->x;
        int nlocal = atom->nlocal;
        for (int i = 0; i < nlocal; i++) {
          x[i][0] = coords[3 * (atom->tag[i] - 1) + 0] * posconv;
          x[i][1] = coords[3 * (atom->tag[i] - 1) + 1] * posconv;
          x[i][2] = coords[3 * (atom->tag[i] - 1) + 2] * posconv;
        }

        if (force->kspace && domain->box_change) { force->kspace->setup(); }

        // compute PE. makes sure that it will be evaluated at next step
        modify->compute[modify->find_compute("thermo_pe")]->invoked_scalar = -1;
        modify->addstep_compute_all(update->ntimestep + 1);
        return;

      } else if (msg.name == "DeleteResults") {
        std::string title;
        pipe->extract_DeleteResults(msg, title);

        if (keptResults.erase(title) == 0) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "DeleteResults", "title",
                               "DeleteResults called with title that was never stored");
        }
        pipe->send_return(AMSPipe::Status::success);    // we are so simple that we never fail ...

      } else {
        throw AMSPipe::Error(AMSPipe::Status::unknown_method, msg.name, "",
                             "unknown method " + msg.name + " called");
      }

    } catch (const AMSPipe::Error &exc) {
      if (msg.name.rfind("Set", 0) == 0) {
        // Exception thrown during "Set" method: buffer it for return later.
        if (!error) error.reset(new AMSPipe::Error(exc));
      } else {
        // Exception thrown during non-"Set" method: return error immediately.
        pipe->send_return(exc.status, exc.method, exc.argument, exc.what());
      }
    }
  }
}

void FixAMSPipe::final_integrate()
{
  if (exiting) {
    timer->force_timeout();
    return;
  }

  AMSPipe::Results results;
  results.energy = modify->compute[modify->find_compute("thermo_pe")]->compute_scalar() * potconv;

  gradients.resize(atom->natoms * 3);
  results.gradients = gradients.data();
  results.gradients_dim[0] = 3;
  results.gradients_dim[1] = atom->natoms;
  double **f = atom->f;

  // reassembles the force vector from the local arrays
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    results.gradients[3 * (atom->tag[i] - 1) + 0] = f[i][0] * gradconv;
    results.gradients[3 * (atom->tag[i] - 1) + 1] = f[i][1] * gradconv;
    results.gradients[3 * (atom->tag[i] - 1) + 2] = f[i][2] * gradconv;
  }
  MPI_Reduce(MPI_IN_PLACE, results.gradients, gradients.size(), MPI_DOUBLE, MPI_SUM, 0, world);

  std::vector<double> stressTensor(9);

  Compute *pressCompute = modify->compute[modify->find_compute("amspipe_pressure")];
  pressCompute->compute_vector();
  stressTensor[0] = pressCompute->vector[0] * stressconv;
  stressTensor[1] = pressCompute->vector[3] * stressconv;
  stressTensor[2] = pressCompute->vector[4] * stressconv;
  stressTensor[3] = stressTensor[1];
  stressTensor[4] = pressCompute->vector[1] * stressconv;
  stressTensor[5] = pressCompute->vector[5] * stressconv;
  stressTensor[6] = stressTensor[2];
  stressTensor[7] = stressTensor[5];
  stressTensor[8] = pressCompute->vector[2] * stressconv;

  results.stressTensor = stressTensor.data();
  results.stressTensor_dim[0] = 3;
  results.stressTensor_dim[1] = 3;

  pipe->send_results(results);
  // we are so simple that we never fail ...
  pipe->send_return(AMSPipe::Status::success);

  int nVectors = latticeVectors.size() / 3;
  if (nVectors == 0) return;

  double **x = atom->x;
  double invPosConv = 1 / posconv;
  for (int i = 0; i < nlocal; i++) {
    coords[3 * (atom->tag[i] - 1) + 0] = x[i][0] * invPosConv;
    coords[3 * (atom->tag[i] - 1) + 1] = x[i][1] * invPosConv;
    coords[3 * (atom->tag[i] - 1) + 2] = x[i][2] * invPosConv;
  }

  prevFrac.resize(coords.size());
  for (int i = 0; i < coords.size(); i += 3) {
    for (int d = 0; d < nVectors; d++) {
      prevFrac[i + d] = coords[i + d] / latticeVectors[3 * d + d];
    }
  }
}

void FixAMSPipe::pre_exchange(void)
{
  // ensure atoms are in current box & update box via shrink-wrap
  // has to be be done before invoking Irregular::migrate_atoms()
  //   since it requires atoms be inside simulation box

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  if (domain->triclinic) domain->lamda2x(atom->nlocal);

  // move atoms to new processors via irregular()
  // only needed if migrate_check() says an atom moves to far
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  if (irregular->migrate_check()) irregular->migrate_atoms();
  if (domain->triclinic) domain->lamda2x(atom->nlocal);
}
