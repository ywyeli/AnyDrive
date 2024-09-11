import xml.etree.ElementTree as ET
from xml.dom import minidom

# Path to your original XML file
file_path = 'routes_training_old.xml'

# Read the original XML from a file
with open(file_path, 'r', encoding='UTF-8') as file:
    original_xml = file.read()

tree = ET.ElementTree(ET.fromstring(original_xml))
root = tree.getroot()

# Create a new root for the reorganized XML
new_root = ET.Element('routes')

# Loop through each route in the original XML
for route in root.findall('route'):
    new_route = ET.SubElement(new_root, 'route', attrib=route.attrib)

    # Create and populate 'weathers' element
    weathers = ET.SubElement(new_route, 'weathers')
    # Add weather elements here...

    # Create and populate 'waypoints' element
    waypoints = ET.SubElement(new_route, 'waypoints')
    for waypoint in route.findall('waypoint'):
        position = ET.SubElement(waypoints, 'position', attrib=waypoint.attrib)

    # Add scenarios here...

# Convert the new tree to a string
new_tree = ET.ElementTree(new_root)
xml_str = ET.tostring(new_root, encoding='unicode')

# Pretty-print the XML
formatted_xml = minidom.parseString(xml_str).toprettyxml(indent="   ")

# Save the pretty-printed XML to a file
with open('routes_reorganized.xml', 'w', encoding='UTF-8') as file:
    file.write(formatted_xml)
