# ActSoft public-syntax fixture only. Not a calibrated Sentaurus project.
#define GATE_STOP 1.2
set DRAIN_BIAS 0.05

File {
  Grid="@tdr@"
  Plot="@tdrdat@"
  Current="@plot@"
}

Electrode {
  { Name="source" Voltage=0.0 }
  { Name="drain" Voltage=@DRAIN_BIAS@ }
  { Name="gate" Voltage=0.0 }
  { Name="body" Voltage=0.0 }
}

Physics {
  Mobility( DopingDep Enormal HighFieldSaturation )
  Recombination( SRH Auger )
  EffectiveIntrinsicDensity( OldSlotboom )
}

Plot(Collected) {
  eDensity hDensity
  eCurrent hCurrent
  ElectricField Potential
}

Math {
  Extrapolate
  Digits=5
  Iterations=30
}

Solve {
  Coupled { Poisson Electron Hole }
  Quasistationary(
    InitialStep=1e-3
    MaxStep=0.05
    Goal { Name="gate" Voltage=@GATE_STOP@ }
  ) {
    Coupled { Poisson Electron Hole }
  }
}

