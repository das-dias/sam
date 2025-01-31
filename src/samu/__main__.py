__doc__ = """Usage: samu.py [-vgrh] FILE

Arguments:
  FILE  input file

Options:
  -h --help
  -v  verbose mode
  -r  extract resistance
  -g  gui mode

"""

import click

from pathlib import Path
from typing import Type, Dict, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, AnyUrl, ConfigDict, field_validator
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *


def yaml_parsable(cls: Type[BaseModel]):
    def to_yaml(self: BaseModel) -> str:
        return to_yaml_str(self)

    def from_yaml(self: Type[BaseModel], yaml_str: str) -> BaseModel:
        return parse_yaml_raw_as(self, yaml_str)

    setattr(cls, to_yaml.__name__, to_yaml)
    setattr(cls, from_yaml.__name__, from_yaml)

    return cls


Unit = Tuple[float, str]


class Units(Enum):
    MICRO = (1e-6, "micro")
    NANO = (1e-9, "nano")
    PICO = (1e-12, "pico")
    FEMTO = (1e-15, "femto")
    
    def __init__(self, scale: float, unit_name: str):
        self.scale = scale
        self.unit_name = unit_name

    def __str__(self):
        return self.unit_name
    

class MaterialType(Enum):
    METAL = "metal"
    DIELECTRIC = "dielectric"


@yaml_parsable
class Material(BaseModel):
    name: str
    material_type: MaterialType
    properties: Dict[str, float]

    @field_validator("material_type", mode="before")
    @classmethod
    def validate_material_type(cls, v):
        if isinstance(v, str):
            return MaterialType(v)
        return v

    @field_validator("properties", mode="before")
    @classmethod
    def validate_properties(cls, v, values):
        if isinstance(v, dict):
            return v
        return {}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        material_type = data.get("material_type", None)
        properties = {k: v for k, v in data.items() if k not in ["material_type", "name"]}
        return cls(name=data.get("name", "Unnamed"), material_type=material_type, properties=properties)


    def __getitem__(self, material_property_name: str) -> float | None:
        return self.properties.get(material_property_name, None)

    def __setitem__(self, material_property_name: str, material_property: float):
        self.properties[material_property_name] = material_property

    def __delitem__(self, material_property_name: str):
        del self.properties[material_property_name]
    
    def __iter__(self):
        return self.properties.__iter__()

@yaml_parsable
class MaterialsDict(BaseModel):
    materials: Dict[str, Material]

    def check(self) -> bool:
        material_types = [t.material_type.value for t in self.materials.values()]
        if len(self.materials) > 1:
            return "metal" in material_types and "dielectric" in material_types
        return "dielectric" in material_types

    def __len__(self) -> int:
        return len(self.materials)
    
    def __getitem__(self, material_name: str) -> Material | None:
        return self.materials.get(material_name, None)

    def __setitem__(self, material_name: str, material: Material):
        self.materials[material_name] = material

    def __delitem__(self, material_name: str):
        del self.materials[material_name]
    
    def __contains__(self, material_name: str):
        return  material_name in self.materials
      
    def __iter__(self):
        return self.materials.__iter__()


Vector = Tuple[float, float, float]


def _draw_box_on_ax(
    ax,
    dimensions: Vector,
    origin: Vector = (0.0, 0.0, 0.0),
    color: str = "gray",
    label: str = "",
):
    # courtesy of ChatGPT - I don't know matplotlib's API
    x, y, z = origin
    length, width, thick = dimensions
    vertices = [
        [x, y, z],
        [x + length, y, z],
        [x + length, y + width, z],
        [x, y + width, z],  # Bottom face
        [x, y, z + thick],
        [x + length, y, z + thick],
        [x + length, y + width, z + thick],
        [x, y + width, z + thick],  # Top face
    ]
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],  # Bottom
        [vertices[i] for i in [4, 5, 6, 7]],  # Top
        [vertices[i] for i in [0, 1, 5, 4]],  # Front
        [vertices[i] for i in [2, 3, 7, 6]],  # Back
        [vertices[i] for i in [0, 3, 7, 4]],  # Left
        [vertices[i] for i in [1, 2, 6, 5]],  # Right
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces, facecolors=color, linewidths=1, edgecolors="k", alpha=0.6
        )
    )
    # Label at the center of the box
    ax.text(
        x + length / 2,
        y + width / 2,
        z + thick / 2,
        label,
        color="black",
        fontsize=12,
        ha="center",
        va="center",
    )


@yaml_parsable
class Geometry(BaseModel):
    units: Unit | None = None
    metal_width: float
    metal_thickness: float
    separation: float
    strip_length: float

    @field_validator("units", mode="before")
    @classmethod
    def validate_units(cls, v):
        if isinstance(v, str):
            assert v in [unit.unit_name for unit in Units], ValueError(f"{v} not in Units: {[unit.unit_name for unit in Units]}")
            return Units[v.upper()].value
        return None
    
    def show(self):
        attacker_origin: Vector = (0.0, 0.0, 0.0)
        victim_origin: Vector = (0.0, self.metal_width + self.separation, 0.0)
        dimensions: Vector = (2*(self.metal_width + self.separation), self.metal_width, self.metal_thickness)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        _draw_box_on_ax(ax, dimensions, attacker_origin, "red", "Attacker")
        _draw_box_on_ax(ax, dimensions, victim_origin, "green", "Victim")
        ax.set_xlabel("X" + self.units[1])
        ax.set_ylabel("Y" + self.units[1])
        ax.set_zlabel("Z" + self.units[1])
        ax.set_xlim(0, 2*(self.metal_width + self.separation))
        ax.set_ylim(0, 2*(self.metal_width + self.separation))
        ax.set_zlim(0, 2*(self.metal_width + self.separation))
        plt.show()


@yaml_parsable
class Result(BaseModel):
    value: float = 0.0
    unit: Units

    def __repr__(self) -> str:
        return f"{self.value / self.unit.value[0]} {self.unit.value[1]}"


CapacitanceResult = Type[Result]
InductanceResult = Type[Result]
ResistanceResult = Type[Result]

class StripLineTModel(SubCircuitFactory):
    NAME = "strip_line_t_model"
    NODES = ("in", "tsec", "gnd", "out")

    def __init__(self, Rs=1 @u_Ω, Ls=0 @u_H, Cs=1e-18 @u_F):
        super().__init__()
        self.R(1, "in", "n1", Rs / 2)
        self.L(1, "n1", "tsec", Ls / 2)
        self.L(2, "tsec", "n2", Ls / 2)
        self.R(2, "n2", "out", Rs / 2)
        self.C(1, "tsec", "gnd", Cs)


class AttackerVictimCrossTalkModel(SubCircuitFactory):
    NAME = "crosstalk_model"
    NODES = ("in", "attacker", "victim", "gnd")

    def __init__(self, Rs=1 @u_Ω, Ls=0 @u_H, Cs=1e-18 @u_F, Cp=1e-18 @u_F, Lp=0 @u_H):
        super().__init__()
        self.subcircuit(StripLineTModel(Rs, Ls, Cs))
        self.X("attacker_circuit", "strip_line_t_model", "in", "attacker", "gnd", "gnd")
        self.X("victim_circuit", "strip_line_t_model", "gnd", "victim", "gnd", "gnd")
        self.C(1, "attacker", "victim", Cp)
        self.L(1, "attacker", "victim", Lp)

@yaml_parsable
class TransientSimulationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    v1: float = 0 @ u_V
    v2: float = 1 @ u_V
    t1: float = 1 @ u_ns
    t2: float = 10 @ u_us
    temperature: float = 25
    nominal_temperature: float = 25
    step_time: float = 1e-11  # s
    end_time: float = 1e-6  # s


_default_transient_sim_config = TransientSimulationConfig()


@yaml_parsable
class Results(BaseModel):
    resistance: ResistanceResult = 0.0
    self_inductance: InductanceResult = 0.0
    mutual_inductance: InductanceResult = 0.0
    mutual_capacitance: CapacitanceResult = 0.0
    ground_capacitance: CapacitanceResult = 0.0

    def show(
        self, sim_config: TransientSimulationConfig = _default_transient_sim_config
    ):
        from numpy import mean, sqrt, log10
        from rich.pretty import pprint
        import PySpice.Logging.Logging as Logging
        logger = Logging.setup_logging()
        pprint(_default_transient_sim_config)
        testbench = Circuit("Crosstalk Testbench")
        testbench.PulseVoltageSource(
            "pulse",
            "input",
            testbench.gnd,
            sim_config.v1,
            sim_config.v2,
            sim_config.t1,
            sim_config.t2,
        )
        testbench.subcircuit(
            AttackerVictimCrossTalkModel(
                Rs=self.resistance,
                Ls=self.self_inductance,
                Cs=self.ground_capacitance,
                Cp=self.mutual_capacitance,
                Lp=self.mutual_inductance,
            )
        )
        testbench.X(
            "xtalk", "crosstalk_model", "input", "attacker", "victim", testbench.gnd
        )

        simulator = testbench.simulator(
            temperature=sim_config.temperature,
            nominal_temperature=sim_config.nominal_temperature,
        )
        analysis = simulator.transient(
            step_time=sim_config.step_time, end_time=sim_config.end_time
        )

        #rms_attacker = sqrt(mean(analysis["attacker"]**2))
        rms_victim = sqrt(mean(analysis["victim"]**2))
        rms_input = sqrt(mean(analysis["input"]**2))
        
        #cross_talk = 10*log10(rms_victim / rms_attacker)
        cross_talk_input = 20*log10(rms_victim / rms_input)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(analysis["input"], color="black")
        ax.plot(analysis["attacker"], color="red")
        ax.plot(analysis["victim"], color="green")
        ax.set_title(f"Crosstalk = 20*log10(RMS[victim] / RMS[input]) = {cross_talk_input:.1f} dB")
        ax.set_xlabel("Timesamples [s]")
        ax.set_ylabel("Voltage (V)")
        ax.set_xlim(0, 1000)
        ax.grid()
        ax.legend(["input", "attacker", "victim"], loc="upper right")
        fig.tight_layout()
        
        



UrlPath = Path | AnyUrl


class Extractor25D(BaseModel):
    materials: MaterialsDict | None = None
    geometry: Geometry | None = None
    results: Results | None = None

    def setup(self, file_path: UrlPath) -> "Extractor25D":
        with open(file_path, "r") as ymlfp:
            def concat_dict(d1, d2):
              d1.update(d2)
              return d1
            
            setup_yaml = yaml.safe_load(ymlfp)
            self.materials = MaterialsDict(
                materials={
                    matname: Material.from_dict(concat_dict(mat, {'name': matname}))
                    for matname, mat in setup_yaml["materials"].items()
                }
            )
            assert len(self.materials) > 0, ValueError(
                "At least a dielectric material is required for the extraction."
            )
            assert len(self.materials) < 3, ValueError(
                "At maximum, 2 materials are required: a metal and an emersing dielectric."
            )
            assert self.materials.check(), ValueError(
                "Exactly 1 dielectric and 1 metal are required for extracting more than capacitive parasitics."
            )
            self.geometry = Geometry(**setup_yaml["geometry"])
        return self

    def extract(self, extract_resistance: bool = False) -> Dict[str, Result]:
        from scipy.constants import mu_0, epsilon_0, pi
        from numpy import log, sqrt
        # based on https://www.emisoftware.com/calculator/ - it has 7% error due to the negligence of thickness in face of width and length
        # also based on: https://ieeexplore.ieee.org/document/328861
        dieletric_name: str = [
            mn
            for mn in self.materials
            if self.materials[mn].material_type == MaterialType.DIELECTRIC
        ][0]
        metal_name: str = [
            mn
            for mn in self.materials
            if self.materials[mn].material_type == MaterialType.METAL
        ][0]
        scale = self.geometry.units[0]
        separation: float = self.geometry.separation * scale
        width: float = self.geometry.metal_width * scale
        thickness: float = self.geometry.metal_thickness * scale
        length: float = self.geometry.strip_length * scale

        #rel_permeability = self.materials[dieletric_name]["rel_permeability"]
        rel_permitivity = self.materials[dieletric_name]["rel_permittivity"]
        
        c = 1 / sqrt(mu_0 * epsilon_0)
        print(c)
        aspect_number = separation / (separation + 2 * width)

        self.results = Results()

        self.results.self_inductance = 0.0

        
        aux1 = 1 - aspect_number**2
        
        if (1/sqrt(2) < aspect_number) and (aspect_number <= 1):            
            self.results.mutual_capacitance = (
                rel_permitivity * length
                / (
                    120 * c
                    * log(-2 / ( (aspect_number**0.5) - 1) * ((aspect_number**0.5) + 1))
                )
            )
            self.results.mutual_inductance = (
                120 * length 
                / c
                * log(2 * (1 + aspect_number**0.5) / (1 - aspect_number**0.5))
            )
        else:
            self.results.mutual_capacitance = (
                rel_permitivity * length / 377 / pi / c
                * log(
                    -2
                    / ((aux1 ** 0.25) - 1)
                    * ((aux1 ** 0.25) + 1)
                )
            )
            self.results.mutual_inductance = (
                377 * pi * length / c
                * log(
                    2
                    * (1 + (aux1 ** 0.25))
                    / (1 - (aux1 ** 0.25))
                )
                
            )
            
        if extract_resistance:
            self.results.resistance = (
                self.materials[metal_name]["resistivity"] * length / (width * thickness)
            )

        return self.results

@click.command()
@click.argument('setup_file')
@click.option('--gui', '-g', is_flag=True, help='GUI mode')
@click.option('--verbose', '-v', is_flag=True, help="verbose mode")
@click.option('--resistance', '-r', is_flag=True, help="extract resistance")
def cli(setup_file, gui, verbose, resistance):
    from rich.pretty import pprint
    from datetime import datetime
    
    assert setup_file, "Provide an input file."
    setup_file = Path(setup_file)
    assert setup_file.exists, FileExistsError(f"{setup_file}")
    print(f"Reading setup file: {setup_file}...")

    extractor = Extractor25D().setup(setup_file)
    results: Results = extractor.extract(resistance)

    if verbose:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"samu > stamp: {current_time} > results:")
        pprint(results)
    print("Done :-)")

    if gui:
        results.show()
        extractor.geometry.show()
        
    exit(0)

if __name__ == "__main__":
    cli()