# S1 BAGEL — NOT VIABLE

`SHARC_BAGEL` inherits from `SHARC_OLD`, which is a legacy abstract base class with no modern class API (`setup_mol / setup_interface / run / getQMout`).
BAGEL is invoked as a standalone CLI script (`python SHARC_BAGEL.py QM.in → QM.out`) and cannot be driven through our embedded-Python bridge.

Same limitation applies to `SHARC_PYSCF`, `SHARC_MOLPRO`, `SHARC_AMS_ADF`, `SHARC_COLUMBUS`.

To add BAGEL support a shell-fork-per-step path would have to be added to `Model_SHARC_Interface::executeKernel_impl`. Not in S1 scope.

See dev log entry #17 (2026-04-25) for full analysis.
