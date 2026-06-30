from __future__ import annotations

import xml.etree.ElementTree as ET

from morphtamp_x_v2.models import PickPlaceScenario
from morphtamp_x_v2.scene_builder import build_pick_place_scene


PANDA_LIKE_XML = """
<mujoco model="panda_like">
  <worldbody>
    <body name="panda_link7">
      <body name="panda_hand"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_build_pick_place_scene_contains_table_cube_target_and_weld(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(PANDA_LIKE_XML, encoding="utf-8")
    output = tmp_path / "scene.xml"

    result = build_pick_place_scene(
        panda_xml=panda,
        scenario=PickPlaceScenario(),
        output_xml=output,
    )

    root = ET.parse(output).getroot()
    names = {item.attrib.get("name") for item in root.iter() if item.attrib.get("name")}

    assert result.output_xml == output.resolve()
    assert "v2_table" in names
    assert "v2_cube" in names
    assert "v2_cube_freejoint" in names
    assert "v2_place_target" in names
    assert "v2_grasp_anchor" in names
    anchor = next(item for item in root.iter("body") if item.attrib.get("name") == "v2_grasp_anchor")
    assert anchor.attrib["mocap"] == "true"
    assert "v2_grasp_weld" in names
    weld = next(item for item in root.iter("weld") if item.attrib.get("name") == "v2_grasp_weld")
    assert weld.attrib["body1"] == "v2_grasp_anchor"
    assert weld.attrib["body2"] == "v2_cube"
    assert weld.attrib["active"] == "false"


def test_build_pick_place_scene_uses_tcp_mocap_anchor_instead_of_robot_body_for_weld(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(
        """
<mujoco model="panda_like">
  <worldbody>
    <body name="link7">
      <body name="hand">
        <body name="attachment"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    output = tmp_path / "scene.xml"

    build_pick_place_scene(
        panda_xml=panda,
        scenario=PickPlaceScenario(),
        output_xml=output,
    )

    root = ET.parse(output).getroot()
    weld = next(item for item in root.iter("weld") if item.attrib.get("name") == "v2_grasp_weld")
    assert weld.attrib["body1"] == "v2_grasp_anchor"


def test_build_pick_place_scene_cube_has_stable_contact_parameters(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(PANDA_LIKE_XML, encoding="utf-8")
    output = tmp_path / "scene.xml"

    build_pick_place_scene(panda_xml=panda, scenario=PickPlaceScenario(), output_xml=output)

    root = ET.parse(output).getroot()
    cube = next(item for item in root.iter("geom") if item.attrib.get("name") == "v2_object_geom")
    assert cube.attrib["condim"] == "4"
    assert cube.attrib["friction"].startswith("1.5")


def test_build_pick_place_scene_absolutizes_mesh_assets(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "dummy.stl"
    mesh.write_text("solid dummy\nendsolid dummy\n", encoding="utf-8")
    panda = tmp_path / "panda.xml"
    panda.write_text(
        """
<mujoco model="mesh_panda">
  <compiler meshdir="assets"/>
  <asset><mesh name="dummy" file="dummy.stl"/></asset>
  <worldbody><body name="panda_hand"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "scene.xml"

    build_pick_place_scene(panda_xml=panda, scenario=PickPlaceScenario(), output_xml=output)

    root = ET.parse(output).getroot()
    mesh_element = next(item for item in root.iter("mesh") if item.attrib.get("name") == "dummy")
    assert mesh_element.attrib["file"] == str(mesh.resolve())


def test_build_pick_place_scene_removes_robot_only_keyframes(tmp_path):
    panda = tmp_path / "panda.xml"
    panda.write_text(
        PANDA_LIKE_XML.replace(
            "</mujoco>",
            '<keyframe><key name="home" qpos="0 0 0"/></keyframe></mujoco>',
        ),
        encoding="utf-8",
    )
    output = tmp_path / "scene.xml"

    build_pick_place_scene(panda_xml=panda, scenario=PickPlaceScenario(), output_xml=output)

    root = ET.parse(output).getroot()
    assert not list(root.iter("keyframe"))
