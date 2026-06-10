# ActSoft public-syntax fixture only. Not a calibrated Sentaurus project.
Device nmos {
  File {
    Grid="@tdr@"
    Plot="@tdrdat@"
    Current="@plot@"
  }
  Electrode {
    { Name="source" Voltage=0.0 }
    { Name="drain" Voltage=0.0 }
    { Name="gate" Voltage=0.0 }
    { Name="body" Voltage=0.0 }
  }
  Physics {
    Mobility( DopingDep Enormal )
    Recombination( SRH )
  }
}

System {
  nmos dut ( "source"=0 "drain"=drain "gate"=gate "body"=0 )
  Vsource_pset vg ( gate 0 ) { dc=0.0 }
  Vsource_pset vd ( drain 0 ) { dc=0.0 }
}

Math {
  Method=Blocked
  SubMethod=ParDiSo
  Iterations=25
}

Solve {
  Coupled { Poisson Electron Hole Contact Circuit }
  Transient(
    InitialTime=0
    FinalTime=1e-9
    InitialStep=1e-12
    MaxStep=1e-10
  ) {
    Coupled { Poisson Electron Hole Contact Circuit }
  }
}

