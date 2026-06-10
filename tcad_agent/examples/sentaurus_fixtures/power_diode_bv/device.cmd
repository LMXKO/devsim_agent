# ActSoft public-syntax fixture only. Not a calibrated Sentaurus project.
set DRIFT_DOPING 1e15
set LIFETIME_SCALE 1.0

File {
  Grid="@tdr@"
  Plot="@tdrdat@"
  Current="@plot@"
  Output="@log@"
}

Electrode {
  { Name="anode" Voltage=0.0 }
  { Name="cathode" Voltage=0.0 }
}

Physics {
  Mobility( DopingDep HighFieldSaturation )
  Recombination( SRH )
  EffectiveIntrinsicDensity( OldSlotboom )
}

Plot {
  eDensity hDensity
  ElectricField Potential
}

Math {
  Extrapolate
  Iterations=20
  NotDamped=50
}

Solve {
  Coupled { Poisson Electron Hole }
  Quasistationary(
    InitialStep=1e-3
    MaxStep=0.1
    MinStep=1e-7
    Goal { Name="cathode" Voltage=-100.0 }
  ) {
    Coupled { Poisson Electron Hole }
  }
}

