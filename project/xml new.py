# import xml.etree.ElementTree as ET
#
# def rearrange_routes(input_file_path, output_file_path, order):
#     """
#     Rearranges routes in an XML file according to a specified order.
#
#     :param input_file_path: Path to the input XML file containing routes.
#     :param output_file_path: Path where the output XML file will be saved.
#     :param order: A list of route IDs in the desired order.
#     """
#     # Parse the XML file
#     tree = ET.parse(input_file_path)
#     root = tree.getroot()
#
#     # Extract routes
#     routes = {int(route.get('id')): route for route in root.findall('route')}
#
#     # Create a new root element for the output XML
#     new_root = ET.Element("routes")
#
#     # Rearrange routes according to the specified order
#     for route_id in order:
#         if route_id in routes:
#             new_root.append(routes[route_id])
#
#     # Write the rearranged routes to a new XML file
#     new_tree = ET.ElementTree(new_root)
#     new_tree.write(output_file_path)
#
# # Example usage of the function
# input_file_path = '/home/ye/Projects/SEG-LiDAR/carla-nuscenes/scenario_runner/srunner/data/routes_training.xml'  # Replace with your input file path
# output_file_path = '/home/ye/Projects/SEG-LiDAR/carla-nuscenes/scenario_runner/srunner/data/routes_nus.xml' # Replace with your desired output file path
# order = [0, 10, 20, 30, 40,
#          1, 11, 21, 31, 41,
#          2, 12, 22, 32, 42,
#          3, 13, 23, 33, 43,
#          4, 14, 24, 34, 44,
#          5, 15, 25, 35, 45,
#          6, 16, 26, 36, 46,
#          7, 17, 27, 37, 47,
#          8, 18, 28, 38, 48,
#          9, 19, 29, 39, 49]  # Replace with the order you want
#
# rearrange_routes(input_file_path, output_file_path, order)