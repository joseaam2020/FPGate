// Global parameters for the FPGate accelerator.
// Everything that depends on the model shape lives here (or in CSRs), never
// as a literal in the RTL. Target model: r7 (see accel/DECISIONES.md).
package fpgate_pkg;

  // GEMM array geometry
  parameter int PE_ROWS = 16;
  parameter int PE_COLS = 16;

  // Datapath widths
  parameter int ACT_W = 9;   // activation into a PE: INT8, or one byte of a 16-bit value (signed 9 bit)
  parameter int WGT_W = 8;   // weights are always INT8
  parameter int ACC_W = 32;  // keep 32: fc2 products reach 22 bits with K up to 2048

  // Latency (cycles) of requant_lane, used by the epilogue to align its sideband signals
  parameter int RQ_LATENCY = 7;

  // Model shape (r7). Informational defaults; the sequencer takes the real values from CSRs.
  parameter int TOKENS = 1025;
  parameter int HEADS  = 8;

endpackage
