# Samu

`A simple coplanar metal stripline electrical cross-talk 2.5D extractor.`

# Install

### Using `pip`

```sh 
$pip install samu
``` 

# Usage

The tool requires two input YAML files:

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

# Contribute

If you want to submit changes, keep it simple. 
Avoid unnecessary comments in the middle of the code - it speaks for itself.
