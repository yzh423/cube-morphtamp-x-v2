from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import xml.etree.ElementTree as ET

from .models import PickPlaceScenario


@dataclass(frozen=True)
class SceneBuildResult:
    output_xml: Path
    panda_xml: Path
    object_body: str = "v2_cube"
    object_joint: str = "v2_cube_freejoint"
    weld_name: str = "v2_grasp_weld"
    target_site: str = "v2_place_target"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_xml": str(self.output_xml),
            "panda_xml": str(self.panda_xml),
            "object_body": self.object_body,
            "object_joint": self.object_joint,
            "weld_name": self.weld_name,
            "target_site": self.target_site,
        }


def _float_text(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(item):.9g}" for item in values)


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _asset_base(root: ET.Element, source_dir: Path, element_tag: str) -> Path:
    preferred = "meshdir" if element_tag == "mesh" else "assetdir"
    for compiler in root.iter("compiler"):
        value = compiler.attrib.get(preferred) or compiler.attrib.get("assetdir")
        if value:
            path = Path(value)
            return path if path.is_absolute() else (source_dir / path)
    return source_dir


def _absolutize_asset_paths(root: ET.Element, source_dir: Path) -> None:
    for element in root.iter():
        filename = element.attrib.get("file")
        if not filename:
            continue
        path = Path(filename)
        if path.is_absolute():
            continue
        element.attrib["file"] = str((_asset_base(root, source_dir, element.tag) / path).resolve())


def _expanded_root(xml_path: Path) -> ET.Element:
    root = copy.deepcopy(ET.parse(xml_path).getroot())
    source_dir = xml_path.parent
    _absolutize_asset_paths(root, source_dir)

    def expand(parent: ET.Element, base_dir: Path) -> None:
        for child in list(parent):
            if child.tag != "include":
                expand(child, base_dir)
                continue
            include_file = child.attrib.get("file")
            if not include_file:
                parent.remove(child)
                continue
            include_path = Path(include_file)
            if not include_path.is_absolute():
                include_path = (base_dir / include_path).resolve()
            included = _expanded_root(include_path)
            index = list(parent).index(child)
            parent.remove(child)
            for included_child in list(included):
                parent.insert(index, copy.deepcopy(included_child))
                index += 1

    expand(root, source_dir)
    return root


def _append_defaults(root: ET.Element) -> None:
    if root.find("option") is None:
        ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    asset = _ensure_child(root, "asset")
    material_names = {item.attrib.get("name") for item in asset.iter("material")}
    for name, rgba in (
        ("v2_table_mat", "0.45 0.45 0.42 1"),
        ("v2_object_mat", "0.05 0.35 1 1"),
        ("v2_target_mat", "0.1 0.9 0.2 0.35"),
        ("v2_obstacle_mat", "0.75 0.25 0.12 1"),
        ("v2_support_mat", "0.36 0.36 0.32 1"),
    ):
        if name not in material_names:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _remove_keyframes(root: ET.Element) -> None:
    for child in list(root):
        if child.tag == "keyframe":
            root.remove(child)


def _choose_weld_body(root: ET.Element) -> str | None:
    names = [body.attrib["name"] for body in root.iter("body") if "name" in body.attrib]
    for token in ("hand", "right_hand", "attachment", "link7"):
        for name in names:
            if token in name.lower() and name != "v2_cube":
                return str(name)
    return None


def build_pick_place_scene(
    *,
    panda_xml: str | Path,
    scenario: PickPlaceScenario,
    output_xml: str | Path,
) -> SceneBuildResult:
    panda_path = Path(panda_xml).resolve()
    output_path = Path(output_xml).resolve()
    if not panda_path.exists():
        raise FileNotFoundError(f"Panda XML does not exist: {panda_path}")
    root = _expanded_root(panda_path)
    root.attrib["model"] = "cube_morphtamp_x_v2_pick_place"
    _append_defaults(root)
    _remove_keyframes(root)
    worldbody = _ensure_child(root, "worldbody")
    table = scenario.table_center
    table_size = tuple(float(item) / 2.0 for item in scenario.table_size)
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "v2_table",
            "type": "box",
            "pos": _float_text(table),
            "size": _float_text(table_size),
            "material": "v2_table_mat",
            "friction": "1.2 0.02 0.002",
            "condim": "4",
        },
    )
    cube_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "v2_cube", "pos": _float_text(scenario.cube_start)},
    )
    ET.SubElement(cube_body, "freejoint", {"name": "v2_cube_freejoint"})
    object_material = next((item for item in root.iter("material") if item.attrib.get("name") == "v2_object_mat"), None)
    if object_material is not None:
        rgba = {
            "cube": "0.05 0.35 1 1",
        "sphere": "0.05 0.75 0.35 1",
        "cylinder": "0.95 0.55 0.08 1",
        "plate": "0.65 0.55 0.95 1",
        "mug_proxy": "0.85 0.30 0.12 1",
        "bowl_proxy": "0.80 0.65 0.25 1",
        "capsule": "0.25 0.75 0.95 1",
        "tall_box": "0.60 0.30 0.90 1",
        "flat_box": "0.20 0.85 0.85 1",
        "ring": "0.95 0.85 0.10 1",
        }.get(scenario.object_type)
        if rgba is not None:
            object_material.attrib["rgba"] = rgba
    ET.SubElement(
        cube_body,
        "geom",
        {
            "name": "v2_object_geom",
            "type": scenario.object_geom_type,
            "size": _float_text(tuple(float(item) for item in scenario.object_size)),
            "mass": f"{float(scenario.object_mass):.9g}",
            "material": "v2_object_mat",
            "friction": "1.5 0.02 0.002",
            "condim": "4",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
        },
    )
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "v2_place_target",
            "type": "sphere",
            "pos": _float_text(scenario.place_target),
            "size": f"{max(float(item) for item in scenario.object_size):.9g}",
            "material": "v2_target_mat",
        },
    )
    if scenario.obstacle_center is not None and scenario.obstacle_size is not None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "v2_obstacle",
                "type": "box",
                "pos": _float_text(scenario.obstacle_center),
                "size": _float_text(tuple(float(item) for item in scenario.obstacle_size)),
                "material": "v2_obstacle_mat",
                "friction": "1.0 0.02 0.002",
                "condim": "4",
            },
        )
    for index, (center, size) in enumerate(scenario.support_blocks):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"v2_support_{index}",
                "type": "box",
                "pos": _float_text(center),
                "size": _float_text(size),
                "material": "v2_support_mat",
                "friction": "1.2 0.02 0.002",
                "condim": "4",
            },
        )
    weld_body = _choose_weld_body(root)
    if weld_body is not None:
        equality = _ensure_child(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            {
                "name": "v2_grasp_weld",
                "body1": weld_body,
                "body2": "v2_cube",
                "active": "false",
                "solref": "0.004 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)
    return SceneBuildResult(output_xml=output_path, panda_xml=panda_path)
