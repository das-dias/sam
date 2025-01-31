# Samu

`A simple coplanar metal stripline electrical cross-talk 2.5D extractor.`

# Install

### PySpice and NgSpice

This tool's GUI relies on PySpice Python lib. and the NgSpice open source SPICE simulator for generating accurate lumped-element circuit model simulations of the channel crosstalk between coplanar strips.

For that, it is required to install [NgSpice](https://pyspice.fabrice-salvaire.fr/releases/v1.5/installation.html). 

On MacOS using Homebrew:

```sh
brew install libngspice
```

On Linux:
```sh
sudo apt install libngspice0
```

On Windows:
```sh
pyspice-post-installation --install-ngspice-dll
```
and then check installation (because with Windows, you never know...)

```sh
pyspice-post-installation --check-install
```

### Install tool using `pip`

```sh 
pip install samu
``` 

# Usage

The tool requires a single input YAML file:

```sh
samu example_setup.yml -g
```

The input setup file contains a description of the dielectric emersing the coplanar metal strips and the geometry of the setup:

```yaml
#example_setup.yml

materials:
  poliomide:
    material_type: dielectric
    rel_permittivity: 3.3
    rel_permeability: 1.0

  copper:
    material_type: metal
    resistivity: 1.68e-8 # Ohm.m

geometry:
  units: micro
  metal_width: 3.0
  metal_thickness: 0.5
  separation: 1.0
  strip_length: 3e4 # 3 cm
```

The `-g`flag enables visualization of the geometry and the crosstalk results using a GUI.

# Visualize results:

In the console:

```sh
Reading setup file: data/example_setup.yml...

samu > stamp: 2025-01-31 03:58:14 > results:
Results(
│   resistance=336.0,
│   self_inductance=0.0,
│   mutual_inductance=7.886464596170601e-07,
│   mutual_capacitance=1.8552987505740626e-12,
│   ground_capacitance=0.0
)
Done :-)
TransientSimulationConfig(
│   v1=UnitValue(0 V),
│   v2=UnitValue(1 V),
│   t1=PeriodValue(1 ns),
│   t2=PeriodValue(10 μs),
│   temperature=25,
│   nominal_temperature=25,
│   step_time=1e-11,
│   end_time=1e-06
)
```

Using the matplotlib-based GUI:

<div align=center>
  <img src="./data/example_gui.png"/>
</div>

# Contribute

If you want to submit changes, keep it simple. 
Avoid unnecessary comments in the middle of the code - it speaks for itself.
