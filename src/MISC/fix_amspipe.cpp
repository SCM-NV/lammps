// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

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
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/******************************************************************************************
 * This fix enables LAMMPS to serve as an AMSPipe worker.
 ******************************************************************************************/

FixAMSPipe::FixAMSPipe(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), irregular(new Irregular(lmp)), pipe(nullptr)
{
  if (strcmp(style,"amspipe") != 0 && narg != 3)
    error->all(FLERR,"Illegal fix amspipe command");

  if (strcmp(arg[1],"all") != 0)
    error->warning(FLERR,"Fix amspipe always uses group all");

  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use fix amspipe without atom IDs");

  if (atom->tag_consecutive() == 0)
    error->all(FLERR,"Fix amspipe requires consecutive atom IDs");

  modify->add_compute("amspipe_pressure all pressure NULL virial");

  // conversions from LAMMPS units to atomic units, which are used by AMSPipe
  potconv=3.1668152e-06/force->boltz;
  posconv=0.52917721*force->angstrom;
  posconv3=posconv*posconv*posconv;
  gradconv = -1 * potconv * posconv;
  stressconv = -1/force->nktv2p*potconv*posconv3;

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
    pipe = new AMSPipe();
  }

  // asks for evaluation of PE at first step
  modify->compute[modify->find_compute("thermo_pe")]->invoked_scalar = -1;
  modify->addstep_compute_all(update->ntimestep + 1);
}

void FixAMSPipe::initial_integrate(int /*vflag*/)
{
  // Variable to store the error until we send the corresponding return message:
  std::unique_ptr<AMSPipe::Error> error;

  while (true) {
    auto msg = pipe->receive();
    // fprintf(stderr, "Method called: %s\n", msg.name.c_str());

    try {
      if (msg.name == "Exit") {
        this->error->done(0);
        break;

      } else if (error) { // We still have an error buffered
        if (msg.name.rfind("Set", 0) == 0) {
          // Calls to "Set" methods are ignored while an error is buffered.
          continue;
        } else {
          // Non-"Set" method called: return buffered error and clear it.
          pipe->send_return(error->status, error->method, error->argument, error->what());
          error.reset();
        }

      } else if (msg.name == "Hello") {
        int64_t version;
        pipe->extract_Hello(msg, version);
        pipe->send_return( version == 1 ? AMSPipe::Status::success : AMSPipe::Status::unknown_version);

      } else if (msg.name == "SetCoords") {
        pipe->extract_SetCoords(msg, coords.data());

      } else if (msg.name == "SetLattice") {
        pipe->extract_SetLattice(msg, latticeVectors);

      } else if (msg.name == "SetSystem") {
        std::vector<std::string> prevAtomSymbols = std::move(atomSymbols);
        pipe->extract_SetSystem(msg, atomSymbols, coords, latticeVectors, totalCharge, bonds, bondOrders, atomicInfo);

        if (!prevAtomSymbols.empty() && atomSymbols != prevAtomSymbols) {
          //FIXME: Reinitialize LAMMPS in a saner way
          std::exit(0);
        }

      } else if (msg.name == "Solve") {
        AMSPipe::SolveRequest request;
        bool keepResults;
        std::string prevTitle;

        pipe->extract_Solve(msg, request, keepResults, prevTitle);

        //std::cout << "Request:" << std::endl;
        //std::cout << "   title: " << request.title << std::endl;
        //std::cout << "   gradients: " << request.gradients << std::endl;
        //std::cout << "   stressTensor: " << request.stressTensor  << std::endl;
        //std::cout << "   elasticTensor: " << request.elasticTensor  << std::endl;
        //std::cout << "   hessian: " << request.hessian  << std::endl;
        //std::cout << "   dipoleMoment: " << request.dipoleMoment  << std::endl;
        //std::cout << "   dipoleGradients: " << request.dipoleGradients  << std::endl;
        //std::cout << "   other: " << request.other  << std::endl;
        //std::cout << "keepResults: " << keepResults << std::endl;
        //if (!prevTitle.empty()) std::cout << "prevTitle: " << prevTitle << std::endl;

        if (keptResults.find(request.title) != keptResults.end()) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "Solve", "title",
                               "title in request corresponds to an already stored results object");
        }
        if (!prevTitle.empty() && keptResults.find(prevTitle) == keptResults.end()) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "Solve", "prevTitle",
                               "prevTitle does not correspond to a kept results object");
        }
        if (keepResults) keptResults.insert(request.title);

        for (int d = 0; d < 3; d++) {
          domain->boxlo[d] = 0;
          domain->boxhi[d] = latticeVectors[3*d+d] * posconv;
        }

        // do error checks on simulation box and set small for triclinic boxes
        domain->set_initial_box();
        // reset global and local box using the new box dimensions
        domain->reset_box();
        // signal that the box has (or may have) changed
        domain->box_change = 1;

        std::vector<double> fractional(coords.size());
        for (int i = 0; i < coords.size(); i += 3) {
          for (int d = 0; d < 3; d++) {
            fractional[i+d] = coords[i+d] / latticeVectors[3*d+d];
          }
        }
        std::vector<int> shift(coords.size());
        if (prevFrac.size() == coords.size()) {
          for (int i = 0; i < coords.size(); i++) {
            shift[i] = std::round(fractional[i] - prevFrac[i]);
          }
        } else {
          prevFrac.clear();
          for (int i = 0; i < coords.size(); i++) {
            shift[i] = std::floor(fractional[i]);
          }
        }
        for (int i = 0; i < coords.size(); i++) {
          fractional[i] -= shift[i];
        }
        for (int i = 0; i < coords.size(); i += 3) {
          for (int d = 0; d < 3; d++) {
            coords[i+d] = fractional[i+d] * latticeVectors[3*d+d];
          }
        }

        double **x = atom->x;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        if (igroup == atom->firstgroup) nlocal = atom->nfirst;
        for (int i = 0; i < nlocal; i++) {
          if (mask[i] & groupbit) {
            x[i][0]=coords[3*(atom->tag[i]-1)+0]*posconv;
            x[i][1]=coords[3*(atom->tag[i]-1)+1]*posconv;
            x[i][2]=coords[3*(atom->tag[i]-1)+2]*posconv;
          }
        }

#if 0
  // check if kspace solver is used
  if (reset_flag && kspace_flag) {
    // reset kspace, pair, angles, ... b/c simulation box might have changed.
    //   kspace->setup() is in some cases not enough since, e.g., g_ewald needs
    //   to be reestimated due to changes in box dimensions.
    force->init();
    // setup_grid() is necessary for pppm since init() is not calling
    //   setup() nor setup_grid() upon calling init().
    if (force->kspace->pppmflag) force->kspace->setup_grid();
    // other kspace styles might need too another setup()?
  } else if (!reset_flag && kspace_flag) {
    // original version
    force->kspace->setup();
  }
#endif
        if (force->kspace) {
          force->kspace->setup();
        }

        // compute PE. makes sure that it will be evaluated at next step
        modify->compute[modify->find_compute("thermo_pe")]->invoked_scalar = -1;
        modify->addstep_compute_all(update->ntimestep+1);
        return;

      } else if (msg.name == "DeleteResults") {
        std::string title;
        pipe->extract_DeleteResults(msg, title);
        //std::cout << "DeleteResults title: " << title << std::endl;

        if (keptResults.erase(title) == 0) {
          throw AMSPipe::Error(AMSPipe::Status::logic_error, "DeleteResults", "title",
                               "DeleteResults called with title that was never stored");
        }
        pipe->send_return(AMSPipe::Status::success); // we are so simple that we never fail ...

      } else {
        throw AMSPipe::Error(AMSPipe::Status::unknown_method, msg.name, "", "unknown method "+msg.name+" called");
      }

    } catch (const AMSPipe::Error& exc) {
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

  AMSPipe::Results results;
  results.energy = modify->compute[modify->find_compute("thermo_pe")]->compute_scalar() * potconv;

  gradients.resize(atom->natoms * 3);
  results.gradients = gradients.data();
  results.gradients_dim[0] = 3;
  results.gradients_dim[1] = atom->natoms;
  double **f= atom->f;

  // reassembles the force vector from the local arrays
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  for (int i = 0; i < nlocal; i++) {
    results.gradients[3*(atom->tag[i]-1)+0] = f[i][0] * gradconv;
    results.gradients[3*(atom->tag[i]-1)+1] = f[i][1] * gradconv;
    results.gradients[3*(atom->tag[i]-1)+2] = f[i][2] * gradconv;
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


  if (true) { // we are so simple that we never fail ...
    pipe->send_results(results);
    pipe->send_return(AMSPipe::Status::success);
  } else { // ... but if we did, we'd send a runtime_error as the return code
    pipe->send_return(AMSPipe::Status::runtime_error, "Solve", "", "error evaluating the potential");
  }

  double **x = atom->x;
  int *mask = atom->mask;
  double invPosConv = 1 / posconv;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      coords[3*(atom->tag[i]-1)+0] = x[i][0] * invPosConv;
      coords[3*(atom->tag[i]-1)+1] = x[i][1] * invPosConv;
      coords[3*(atom->tag[i]-1)+2] = x[i][2] * invPosConv;
    }
  }

  prevFrac.resize(coords.size());
  for (int i = 0; i < coords.size(); i += 3) {
    for (int d = 0; d < 3; d++) {
      prevFrac[i+d] = coords[i+d] / latticeVectors[3*d+d];
    }
  }
  prevCoord = coords;

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


