"""
A simple coplanar metal stripline electrical cross-talk 2.5D extractor.

Usage:
  samu <setup.yaml> [-g] [-r] [-v]

Options:
  -g --gui   Enable GUI mode
  -r         Extract resistance
  -h --help  Show this help message
  -v         Show verbose results.
"""

from docopt import docopt
from pathlib import Path
from typing import Type, Dict, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, AnyUrl, ConfigDict
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySpice.Spice.Netlist import Circuit, SubCircuit
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

class MaterialType(Enum):
    METAL = "metal"
    DIELECTRIC = "dielectric"


@yaml_parsable
class Material(BaseModel):
    name: str
    material_type: MaterialType
    properties: Dict[str, float]

    def __init__(
        self,
        name: str,
        material_type: Optional[str] = None,
        properties: Dict[str, float] = {},
        **kwargs,
    ):
        self.name = name
        self.material_type = MaterialType(material_type)
        self.properties = properties
        if kwargs:
            self.material_type = MaterialType(kwargs["material_type"])
            self.properties.update(
                {k: v} for k, v in kwargs.items() if k != "material_type"
            )

    def __get__(self, material_property_name: str) -> float:
        return self.properties[material_property_name]

    def __set__(self, material_property_name: str, material_property: float):
        self.properties[material_property_name] = material_property


@yaml_parsable
class MaterialsDict(BaseModel):
    materials: Dict[str, Material]

    def check(self) -> bool:
        material_types = [t.material_type.value for t in self.materials.values()]
        if len(self.materials) > 1:
            return "metal" in material_types and "dielectric" in material_types
        return "dielectric" in material_types

    def __get__(self, material_name: str) -> Material:
        return self.materials.get(material_name, None)

    def __set__(self, material_name: str, material: Material):
        self.materials[material_name] = material

    def __delete__(self, material_name: str):
        del self.materials[material_name]


Vector = Tuple[float, float, float]


def _draw_box_on_ax(
    ax,
    dimensions: Vector,
    color: str = "gray",
    origin: Vector = (0.0, 0.0, 0.0),
    label: str = "",
):
    # courtesy of ChatGPT - I don't know matplotlib's API
    x, y, z = origin
    width, thick, length = dimensions
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
    units: Unit
    metal_width: float
    metal_thickness: float
    separation: float
    strip_length: float

    def show(self):
        attacker_origin: Vector = (0.0, 0.0, 0.0)
        victim_origin: Vector = (0.0, self.metal_width + self.separation, 0.0)
        dimensions: Vector = (self.strip_length, self.metal_width, self.metal_thickness)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        _draw_box_on_ax(ax, attacker_origin, dimensions, "red", "Attacker")
        _draw_box_on_ax(ax, victim_origin, dimensions, "green", "Victim")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
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


class StripLineTModel(SubCircuit):
    __name__ = "strip_line_t_model"
    __nodes__ = ("in", "tsec", "gnd", "out")

    def __init__(self, Rs=1 @u_Ω, Ls=0 @u_H, Cs=1e-18 @u_F):
        super().__init__()
        self.R(1, "in", "n1", Rs / 2)
        self.L(1, "n1", "tsec", Ls / 2)
        self.L(2, "tsec", "n2", Ls / 2)
        self.R(2, "n2", "out", Rs / 2)
        self.C(1, "tsec", "gnd", Cs)


class AttackerVictimCrossTalkModel(SubCircuit):
    __name__ = "crosstalk_model"
    __nodes__ = ("in", "attacker", "victim", "gnd")

    def __init__(self, Rs=1 @u_Ω, Ls=0 @u_H, Cs=1e-18 @u_F, Cp=1e-18 @u_F, Lp=0 @u_H):
        super().__init__()
        self.subcircuit(StripLineTModel(Rs=Rs, Ls=Ls, Cs=Cs))
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
    end_time: float = 20e-6  # s


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
        import PySpice.Logging.Logging as Logging

        logger = Logging.setup_logging()

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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(analysis["input"], color="black")
        ax.plot(analysis["attacker"], color="red")
        ax.plot(analysis["victim"], color="green")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage (V)")
        ax.grid()
        ax.legend(["input", "attacker", "victim"], loc="upper right")
        fig.tight_layout()


UrlPath = Path | AnyUrl


class Extractor25D(BaseModel):
    materials: MaterialsDict
    geometry: Geometry
    results: Results

    def setup(self, file_path: UrlPath) -> "Extractor25D":
        with open(file_path, "r") as ymlfp:
            setup_yaml = yaml.safe_load(ymlfp)
            self.materials = MaterialsDict(
                materials={
                    matname: Material(name=matname, **mat)
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

    def extract(self, extract_r: bool = False) -> Dict[str, Result]:
        from scipy.constants import mu_0, epsilon_0, pi
        from numpy import log
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
        separation: float = self.geometry.separation
        width: float = self.geometry.metal_width
        thickness: float = self.geometry.metal_thickness
        length: float = self.geometry.strip_length

        rel_permeability = self.materials[dieletric_name]["rel_permeability"]
        rel_permitivity = self.materials[dieletric_name]["rel_permittivity"]

        eps = epsilon_0 * rel_permitivity
        mu = mu_0 * rel_permeability
        c = 1 / (mu * eps)

        aspect_number = separation / (separation + 2 * width)

        second_form: bool = aspect_number <= 2**0.5 / 2

        self.results = Results()

        self.results.self_inductance = 0.0

        if second_form:
            self.results.mutual_capacitance = (
                eps
                * length
                / (
                    120
                    * c
                    * log(-2 / (aspect_number**0.5 - 1) * (aspect_number**0.5 + 1))
                )
            )
            self.results.mutual_inductance = (
                377
                * pi
                * length
                / (
                    c
                    * log(
                        2
                        * (1 + (1 - aspect_number**2) ** 0.25)
                        / (1 - (1 - aspect_number**2) ** 0.25)
                    )
                )
            )

        else:
            self.results.mutual_capacitance = (
                eps
                * length
                / (377 * pi * c)
                * log(
                    -2
                    / ((1 - aspect_number**2) ** 0.25 - 1)
                    * ((1 - aspect_number**2) ** 0.25 + 1)
                )
            )
            self.results.mutual_inductance = (
                120
                * length
                / c
                * log(2 * (1 + aspect_number**0.5) / (1 - aspect_number**0.5))
            )

        if extract_r:
            self.results.resistance = (
                self.materials[metal_name]["resistivity"] * length / (width * thickness)
            )

        return self.results


def main():
    from rich.pretty import pprint
    from datetime import datetime

    arguments = docopt(__doc__)
    setup_file = Path(arguments["<setup.yaml>"])
    enable_gui = arguments["--gui"]
    enable_r = arguments["-r"]
    enable_i = arguments["-i"]
    verbose = arguments["-v"]
    assert setup_file.exists, FileExistsError(f"{setup_file}")
    print(f"Reading setup file: {setup_file}...")

    extractor = Extractor25D().setup(setup_file)
    results: Results = extractor.extract(enable_r, enable_i)

    if verbose:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"sam > stamp: {current_time} > results:")
        pprint(results)
    print("Done :-)")

    if enable_gui:
        results.show()
        extractor.geometry.show()
        
    exit(0)


if __name__ == "__main__":
    main()
